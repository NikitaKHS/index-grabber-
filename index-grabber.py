#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import logging
import argparse
import threading
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple


# -------------------- HTML links --------------------

class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for k, v in attrs:
            if k.lower() == "href" and v:
                self.links.append(v)


def get_links(opener: urllib.request.OpenerDirector, url: str, timeout: int) -> List[str]:
    try:
        req = urllib.request.Request(url, method="GET")
        with opener.open(req, timeout=timeout) as resp:
            data = resp.read()
        html = data.decode("utf-8", errors="ignore")
        p = LinkParser()
        p.feed(html)
        return p.links
    except (HTTPError, URLError) as e:
        logging.warning("Листинг не прочитался %s: %s", url, e)
        return []
    except Exception as e:
        logging.warning("Листинг не прочитался %s: %s", url, e)
        return []


# -------------------- utils --------------------

def get_exe_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def normalize_base_url(url: str) -> str:
    url = url.strip()
    if not url:
        raise ValueError("URL пустой")
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme:
        raise ValueError("Укажи схему, например https://example.com/path/")
    if not url.endswith("/"):
        url += "/"
    return url


def is_junk_href(href: str) -> bool:
    if not href:
        return True
    h = href.strip()
    if h in ("../", "./", "..", "."):
        return True
    low = h.lower()
    if low.startswith(("mailto:", "javascript:", "tel:")):
        return True
    if h.startswith("#"):
        return True
    # autoindex сортировки
    if h.startswith("?"):
        return True
    return False


def safe_join(base_dir: str, rel_url_path: str) -> str:
    base = os.path.abspath(base_dir)

    rel = urllib.parse.unquote(rel_url_path).replace("\\", "/")
    rel = rel.lstrip("/")  # чтобы не стал абсолютным путём
    full = os.path.abspath(os.path.normpath(os.path.join(base, rel)))

    # base + os.sep, чтобы не было совпадений типа C:\down vs C:\downloads2
    if full == base or full.startswith(base + os.sep):
        return full
    raise ValueError("выход за пределы папки")


def make_opener(user_agent: str) -> urllib.request.OpenerDirector:
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", user_agent)]
    return opener


def format_bytes(n: int) -> str:
    # максимально простое, но читабельное
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            if u == "B":
                return f"{int(x)} {u}"
            return f"{x:.1f} {u}"
        x /= 1024
    return f"{n} B"


# -------------------- crawl (collect tasks) --------------------

def crawl_collect(
    opener: urllib.request.OpenerDirector,
    base_url: str,
    timeout: int,
    max_depth: int,
    include_re: Optional[re.Pattern],
    exclude_re: Optional[re.Pattern],
) -> List[Tuple[str, str]]:
    """
    Возвращает список задач: (file_url, rel_url_path)
    rel_url_path — путь внутри base_url, используется для локального пути.
    """
    visited: Set[str] = set()
    tasks: List[Tuple[str, str]] = []

    def walk(dir_url: str, depth: int):
        if dir_url in visited:
            return
        visited.add(dir_url)

        if max_depth >= 0 and depth > max_depth:
            return

        links = get_links(opener, dir_url, timeout=timeout)
        for href in links:
            if is_junk_href(href):
                continue

            child_url = urllib.parse.urljoin(dir_url, href)

            # чтобы не уехать на другой домен/путь
            if not child_url.startswith(base_url):
                continue

            if href.endswith("/"):
                # директорийный URL пусть всегда со слешом
                if not child_url.endswith("/"):
                    child_url += "/"
                walk(child_url, depth + 1)
            else:
                rel = child_url[len(base_url):]
                name = os.path.basename(urllib.parse.unquote(rel))

                if exclude_re and exclude_re.search(name):
                    continue
                if include_re and not include_re.search(name):
                    continue

                tasks.append((child_url, rel))

    walk(base_url, 0)
    return tasks


# -------------------- download (resume + retries) --------------------

class Progress:
    def __init__(self, total_files: int, total_bytes_hint: int) -> None:
        self.total_files = total_files
        self.total_bytes_hint = total_bytes_hint

        self.files_done = 0
        self.bytes_done = 0
        self.files_failed = 0

        self._lock = threading.Lock()
        self._start = time.time()
        self._last_print = 0.0
        self._current_file = ""  # для отображения, без гарантий точности (многопоток)

    def set_current_file(self, s: str) -> None:
        with self._lock:
            self._current_file = s

    def add_bytes(self, n: int) -> None:
        with self._lock:
            self.bytes_done += n

    def file_done(self, ok: bool) -> None:
        with self._lock:
            self.files_done += 1
            if not ok:
                self.files_failed += 1

    def render(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_print) < 0.25:
            return

        with self._lock:
            done = self.files_done
            total = self.total_files
            bdone = self.bytes_done
            bhint = self.total_bytes_hint
            failed = self.files_failed
            cur = self._current_file

        dt = max(now - self._start, 0.001)
        speed = int(bdone / dt)

        if bhint > 0:
            pct = min(100.0, (bdone / bhint) * 100.0)
            bar_w = 24
            fill = int(bar_w * pct / 100.0)
            bar = "█" * fill + "░" * (bar_w - fill)
            left = max(bhint - bdone, 0)
            eta = int(left / max(speed, 1))
            line = (
                f"[{bar}] {pct:5.1f}% | "
                f"{done}/{total} files"
                f"{' | fail ' + str(failed) if failed else ''} | "
                f"{format_bytes(bdone)}/{format_bytes(bhint)} | "
                f"{format_bytes(speed)}/s | ETA {eta}s"
            )
        else:
            line = (
                f"{done}/{total} files"
                f"{' | fail ' + str(failed) if failed else ''} | "
                f"{format_bytes(bdone)} | {format_bytes(speed)}/s"
            )

        if cur:
            # чтобы не расползалось по экрану
            cur_short = (cur[:80] + "…") if len(cur) > 81 else cur
            line += f" | {cur_short}"

        # печать в одну строку
        sys.stdout.write("\r" + line.ljust(160))
        sys.stdout.flush()
        self._last_print = now

    def finish(self) -> None:
        self.render(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()


def head_content_length(opener: urllib.request.OpenerDirector, url: str, timeout: int) -> int:
    """
    Пытаемся взять Content-Length.
    Если сервер не любит HEAD, пробуем GET с Range: bytes=0-0.
    """
    # 1) HEAD
    try:
        req = urllib.request.Request(url, method="HEAD")
        with opener.open(req, timeout=timeout) as resp:
            cl = resp.headers.get("Content-Length")
            if cl and cl.isdigit():
                return int(cl)
    except Exception:
        pass

    # 2) GET bytes=0-0
    try:
        req = urllib.request.Request(url, method="GET", headers={"Range": "bytes=0-0"})
        with opener.open(req, timeout=timeout) as resp:
            cr = resp.headers.get("Content-Range", "")
            # Content-Range: bytes 0-0/12345
            if "/" in cr:
                total = cr.split("/")[-1].strip()
                if total.isdigit():
                    return int(total)
            cl = resp.headers.get("Content-Length")
            if cl and cl.isdigit():
                # это будет 1, но вдруг сервер странный
                return int(cl)
    except Exception:
        pass

    return -1


def download_one(
    opener: urllib.request.OpenerDirector,
    url: str,
    out_dir: str,
    rel: str,
    timeout: int,
    retries: int,
    retry_sleep: float,
    skip_existing: bool,
    progress: Progress,
) -> Tuple[bool, int]:
    """
    Скачивает один файл.
    Возвращает (ok, downloaded_bytes_in_this_call)
    """
    try:
        local_path = safe_join(out_dir, rel)
    except Exception:
        return False, 0

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Если файл уже есть и выглядит целым, пропускаем.
    # "выглядит целым" = совпал размер, если можем узнать.
    remote_len = head_content_length(opener, url, timeout)
    if skip_existing and os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        if remote_len > 0 and os.path.getsize(local_path) == remote_len:
            return True, 0
        if remote_len <= 0:
            # размер не знаем, но файл уже есть -> не трогаем
            return True, 0

    part_path = local_path + ".part"
    existing_part = os.path.getsize(part_path) if os.path.exists(part_path) else 0

    # Если есть .part и он уже равен remote_len, просто допереименуем
    if remote_len > 0 and existing_part == remote_len:
        os.replace(part_path, local_path)
        return True, 0

    attempt = 0
    while True:
        attempt += 1
        try:
            # resume: если есть .part, продолжаем
            headers = {}
            mode = "wb"
            start_from = 0

            if os.path.exists(part_path):
                start_from = os.path.getsize(part_path)
                if start_from > 0:
                    headers["Range"] = f"bytes={start_from}-"
                    mode = "ab"

            progress.set_current_file(os.path.basename(urllib.parse.unquote(rel)))

            req = urllib.request.Request(url, method="GET", headers=headers)
            with opener.open(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)

                # если Range не поддержан и вернули 200 на запрос с Range — начнём заново
                if "Range" in headers and status == 200:
                    try:
                        os.remove(part_path)
                    except Exception:
                        pass
                    start_from = 0
                    mode = "wb"

                chunk_total = 0
                with open(part_path, mode) as f:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
                        chunk_total += len(chunk)
                        progress.add_bytes(len(chunk))
                        progress.render()

            # финальная проверка размера, если знаем
            final_size = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            if remote_len > 0 and final_size != remote_len:
                raise IOError(f"размер не сошёлся: {final_size} != {remote_len}")

            os.replace(part_path, local_path)
            return True, chunk_total

        except HTTPError as e:
            code = getattr(e, "code", None)
            # 4xx обычно не лечится, но 408/429 можно ретраить
            if code and 400 <= code < 500 and code not in (408, 429):
                logging.error("HTTP %s %s", code, url)
                return False, 0
            logging.warning("HTTP %s %s (попытка %d/%d)", code, url, attempt, retries + 1)

        except (URLError, TimeoutError) as e:
            logging.warning("Сеть %s (попытка %d/%d): %s", url, attempt, retries + 1, e)

        except Exception as e:
            logging.warning("Ошибка %s (попытка %d/%d): %s", url, attempt, retries + 1, e)

        if attempt > retries:
            return False, 0
        time.sleep(retry_sleep)


# -------------------- runner --------------------

def run_download(
    base_url: str,
    out_dir: str,
    threads: int,
    timeout: int,
    retries: int,
    retry_sleep: float,
    max_depth: int,
    include: str,
    exclude: str,
    skip_existing: bool,
    user_agent: str,
) -> None:
    opener = make_opener(user_agent)

    include_re = re.compile(include) if include else None
    exclude_re = re.compile(exclude) if exclude else None

    logging.info("Сканирую ссылки... Действие может занять несколько минут.")
    tasks = crawl_collect(opener, base_url, timeout, max_depth, include_re, exclude_re)

    if not tasks:
        print("Ничего не нашёл. Либо пустая папка, либо это не HTML листинг.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # попробуем оценить общий объём (не всегда получится быстро, но обычно ок)
    total_hint = 0
    known = 0
    for url, _rel in tasks:
        n = head_content_length(opener, url, timeout)
        if n > 0:
            total_hint += n
            known += 1

    progress = Progress(total_files=len(tasks), total_bytes_hint=total_hint)
    print(f"Файлов: {len(tasks)}" + (f", объём примерно: {format_bytes(total_hint)}" if known else ""))

    ok_count = 0
    fail_count = 0

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        futures = []
        for url, rel in tasks:
            futures.append(pool.submit(
                download_one,
                opener,
                url,
                out_dir,
                rel,
                timeout,
                retries,
                retry_sleep,
                skip_existing,
                progress,
            ))

        for f in as_completed(futures):
            try:
                ok, _bytes = f.result()
            except Exception:
                ok = False

            if ok:
                ok_count += 1
            else:
                fail_count += 1

            progress.file_done(ok)
            progress.render()

    progress.finish()
    dt = time.time() - t0
    print(f"Готово. OK: {ok_count}, FAIL: {fail_count}, время: {dt:.1f} сек")


# -------------------- CLI + interactive --------------------

def setup_logging(verbose: bool, log_path: str) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        try:
            handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
        except Exception as e:
            print(f"Лог-файл не открылся: {e}", file=sys.stderr)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("url", nargs="?", default="")
    p.add_argument("-o", "--out", default="")
    p.add_argument("-t", "--threads", type=int, default=4)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--retry-sleep", type=float, default=1.0)
    p.add_argument("--max-depth", type=int, default=-1)
    p.add_argument("--include", default="", help="regex по имени файла")
    p.add_argument("--exclude", default="", help="regex по имени файла")
    p.add_argument("--no-skip", action="store_true", help="не пропускать существующие файлы")
    p.add_argument("--ua", default="IndexGrabber/2.0 (+urllib)")
    p.add_argument("--log", default="")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def interactive_if_needed(args) -> Tuple[str, str]:
    url = args.url.strip()
    if not url:
        print("=" * 60)
        print("Скачивание с HTTP index")
        print("=" * 60)
        url = input("URL: ").strip()

    base_url = normalize_base_url(url)

    default_out = os.path.join(get_exe_dir(), "downloads")
    out_dir = args.out.strip()
    if not out_dir:
        out_dir = input(f"Папка [{default_out}]: ").strip() or default_out

    return base_url, out_dir


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(args.verbose, args.log)

    try:
        base_url, out_dir = interactive_if_needed(args)
    except ValueError as e:
        print(f"Ошибка: {e}")
        return 2

    run_download(
        base_url=base_url,
        out_dir=out_dir,
        threads=max(1, args.threads),
        timeout=args.timeout,
        retries=max(0, args.retries),
        retry_sleep=max(0.0, args.retry_sleep),
        max_depth=args.max_depth,
        include=args.include,
        exclude=args.exclude,
        skip_existing=not args.no_skip,
        user_agent=args.ua,
    )

    # чтобы на Windows окно не закрывалось сразу
    if not args.url:
        input("Enter для выхода...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

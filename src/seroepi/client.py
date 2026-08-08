"""Module providing API clients for external serotyping services.

Includes clients for both the Pathogenwatch Next API and the Kaptive-Web
FastAPI service, unified by a fast BaseAPIClient.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
import polars as pl

import io
import joblib
import orjson
import pycountry
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if TYPE_CHECKING:
    pass


class BaseAPIClient:
    """Base API client providing session management, retries, and rapid JSON parsing."""

    def __init__(  # noqa: ANN204, D107
        self,
        base_url: str,
        api_key_header: str | None = None,
        api_key: str | None = None,
        user_agent: str = "seroepi-client/1.0",
    ):
        self._base_url = base_url.rstrip("/")
        self.session = requests.Session()

        headers = {"User-Agent": user_agent, "Content-Type": "application/json"}
        if api_key_header and api_key:
            headers[api_key_header] = api_key

        self.session.headers.update(headers)

        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "DELETE"],
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def __enter__(self):  # noqa: ANN204, D105
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001, ANN204, D105
        self.session.close()

    def _url(self, endpoint: str) -> str:
        return f"{self._base_url}/{endpoint.lstrip('/')}"

    def request(self, method: str, endpoint: str, **kwargs) -> Any:  # noqa: ANN003, ANN401
        """Sends an HTTP request and parses the response with orjson.

        Note: If the response is empty or not valid JSON, it returns the raw text.
        """
        url = self._url(endpoint)

        # Fast outbound serialization
        if "json" in kwargs:
            kwargs["data"] = orjson.dumps(kwargs.pop("json"))

        # Avoid sending application/json header when uploading multipart/form-data
        if "files" in kwargs and self.session.headers.get("Content-Type"):
            old_ct = self.session.headers.pop("Content-Type", None)
            response = self.session.request(method, url, **kwargs)
            if old_ct:
                self.session.headers["Content-Type"] = old_ct
        else:
            response = self.session.request(method, url, **kwargs)

        response.raise_for_status()

        if not response.content:
            return None

        try:
            return orjson.loads(response.content)
        except orjson.JSONDecodeError:
            return response.text

    def get(self, endpoint: str, **kwargs) -> Any:  # noqa: ANN003, ANN401, D102
        return self.request("GET", endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs) -> Any:  # noqa: ANN003, ANN401, D102
        return self.request("POST", endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> Any:  # noqa: ANN003, ANN401, D102
        return self.request("DELETE", endpoint, **kwargs)

    def stream(self, method: str, endpoint: str, **kwargs) -> io.BytesIO:
        """Sends a request and returns the response content wrapped in an io.BytesIO stream."""
        url = self._url(endpoint)
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return io.BytesIO(response.content)


class PathogenwatchClient(BaseAPIClient):
    """Client for the Pathogenwatch Next API."""

    _COLLECTIONS_ENDPOINT = "collections/list"
    _FOLDERS_ENDPOINT = "folders/list"

    def __init__(self, api_key: str):  # noqa: ANN204, D107
        super().__init__(
            base_url="https://next.pathogen.watch/api",
            api_key_header="X-API-Key",
            api_key=api_key,
        )

    def prefetch(self, items: Iterable[PathogenwatchContainerMixin], max_workers: int = 10) -> None:  # noqa: D102
        from joblib import Parallel, delayed
        Parallel(n_jobs=max_workers, require="sharedmem")(
            delayed(item.get_genomes)(self) for item in items
        )

    def get_collections(  # noqa: D102
        self, exclude: str | None = None, limit: int | None = None, binned: bool | None = None
    ) -> Generator[PathogenwatchCollection, None, None]:
        params = {
            k: v
            for k, v in [
                ("exclude", exclude),
                ("limit", limit),
                ("binned", str(binned).lower() if binned is not None else None),
            ]
            if v is not None
        }
        valid_keys = {f.name for f in fields(PathogenwatchCollection) if f.init}

        for collection_dict in self.get(self._COLLECTIONS_ENDPOINT, params=params):
            yield PathogenwatchCollection(**{k: v for k, v in collection_dict.items() if k in valid_keys})

    def get_folders(  # noqa: D102
        self, exclude: str | None = None, limit: int | None = None, binned: bool | None = None
    ) -> Generator[PathogenwatchFolder, None, None]:
        params = {
            k: v
            for k, v in [
                ("exclude", exclude),
                ("limit", limit),
                ("binned", str(binned).lower() if binned is not None else None),
            ]
            if v is not None
        }
        valid_keys = {f.name for f in fields(PathogenwatchFolder) if f.init}

        for folder_dict in self.get(self._FOLDERS_ENDPOINT, params=params):
            yield PathogenwatchFolder(**{k: v for k, v in folder_dict.items() if k in valid_keys})

    def fetch_assembly(self, genome_id: str) -> io.BytesIO:
        """Fetch a genome assembly FASTA as a binary stream."""
        return self.stream("GET", f"genomes/{genome_id}/fasta")


class PathogenwatchContainerMixin:  # noqa: D101
    _ENTITY_TYPE: ClassVar[str]
    _DETAILS_QUERY_PARAM: ClassVar[str]
    _GENOMES_ID_PARAM: ClassVar[str]
    _GENOMES_CURSOR_PARAM: ClassVar[str]
    _ATTR_PREFIX: ClassVar[str]
    uuid: str
    _details_cache: dict[str, Any] | None

    def get_details(self, client: PathogenwatchClient) -> dict[str, Any]:  # noqa: D102
        if self._details_cache is None:
            details = client.get(f"{self._ENTITY_TYPE}/details", params={self._DETAILS_QUERY_PARAM: self.uuid})
            object.__setattr__(self, "_details_cache", details)
        assert self._details_cache is not None
        return self._details_cache

    def get_genomes(self, client: PathogenwatchClient, limit: int = 1000) -> list[dict[str, Any]]:  # noqa: D102
        if self._GENOMES_ID_PARAM == "uuid":
            query_id = self.uuid
        else:
            query_id = self.get_details(client).get("id")
            if not query_id:
                raise ValueError(f"Could not resolve internal ID for {self._ENTITY_TYPE[:-1]} {self.uuid}")

        all_genomes = []
        cursor = None

        while True:
            params = {self._GENOMES_ID_PARAM: query_id, "limit": limit}
            if cursor:
                params[self._GENOMES_CURSOR_PARAM] = cursor

            data = client.get(f"{self._ENTITY_TYPE}/genomes", params=params)
            all_genomes.extend(data.get("genomes", []))

            cursor = data.get("meta", {}).get("endCursor")
            if not cursor or data.get("meta", {}).get("empty"):
                break

        return all_genomes


@dataclass(frozen=True, slots=True)
class PathogenwatchCollection(PathogenwatchContainerMixin):  # noqa: D101
    _ENTITY_TYPE: ClassVar[str] = "collections"
    _DETAILS_QUERY_PARAM: ClassVar[str] = "uuid"
    _GENOMES_ID_PARAM: ClassVar[str] = "uuid"
    _GENOMES_CURSOR_PARAM: ClassVar[str] = "cursor"
    _ATTR_PREFIX: ClassVar[str] = "pw_collection"

    binned: bool
    createdAt: str
    description: str
    name: str
    organismId: str
    owner: str
    uuid: str
    size: int
    _details_cache: dict[str, Any] | None = field(default=None, init=False, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class PathogenwatchFolder(PathogenwatchContainerMixin):  # noqa: D101
    _ENTITY_TYPE: ClassVar[str] = "folders"
    _DETAILS_QUERY_PARAM: ClassVar[str] = "id"
    _GENOMES_ID_PARAM: ClassVar[str] = "folderId"
    _GENOMES_CURSOR_PARAM: ClassVar[str] = "after"
    _ATTR_PREFIX: ClassVar[str] = "pw_folder"

    createdAt: str
    id: str
    uuid: str
    access: str
    name: str = ""
    binned: bool = False
    _details_cache: dict[str, Any] | None = field(default=None, init=False, repr=False, compare=False)


class KaptiveClient(BaseAPIClient):
    """Client for the Kaptive-Web FastAPI service."""

    def __init__(self, base_url: str, api_key: str):  # noqa: ANN204, D107
        super().__init__(
            base_url=f"{base_url.rstrip('/')}/serotype",
            api_key_header="X-API-Key",
            api_key=api_key,
            user_agent="seroepi-kaptive-client/1.0",
        )

    def get_species(self) -> list[str]:  # noqa: D102
        return self.get("species")

    def get_databases(self, species: str) -> list[dict[str, Any]]:  # noqa: D102
        return self.get(f"databases/{species}")

    def get_runs(self) -> list[dict[str, Any]]:  # noqa: D102
        return self.get("runs")

    def get_run_results(self, run_id: str, include_results: bool = True) -> dict[str, Any]:  # noqa: D102
        return self.get(f"runs/{run_id}", params={"include_results": str(include_results).lower()})

    def fetch_parsed_run(self, run_id: str, include_results: bool = True) -> pl.DataFrame:
        """Fetch a run's results and parse them into a Polars DataFrame.

        Uses :class:`KaptiveNativeParser` to turn the JSON payload into a DataFrame
        that matches the internal SeroEpi schema.
        """
        from seroepi.io import KaptiveNativeParser
        run_results = self.get_run_results(run_id, include_results=include_results)
        return KaptiveNativeParser.from_run_results(run_results)

    def fetch_all_parsed_runs(self, include_results: bool = True) -> pl.DataFrame:
        """Fetch *all* runs from the Kaptive-Web service and concatenate their parsed results.

        The method executes in parallel to aggregate the parsed DataFrames.
        If no runs are present an empty DataFrame with a ``sample_id`` column is returned.
        """
        from joblib import Parallel, delayed
        
        runs = self.get_runs()
        run_ids = [run.get("run_id") or run.get("id") for run in runs if run.get("run_id") or run.get("id")]
        
        dfs = Parallel(n_jobs=-1, require="sharedmem")(
            delayed(self.fetch_parsed_run)(run_id, include_results) for run_id in run_ids
        )
        
        if not dfs:
            return pl.DataFrame({"sample_id": []})
        return pl.concat(dfs, how="vertical")


    def get_all_results(self) -> list[dict[str, Any]]:  # noqa: D102
        return self.get("results")

    def submit_job(  # noqa: D102
        self,
        species: str,
        fasta_paths: list[str | Path],
        run_name: str | None = None,
    ) -> dict[str, Any]:
        files_payload = []
        file_handles = []

        try:
            for p in fasta_paths:
                path = Path(p)
                if not path.exists():
                    raise FileNotFoundError(f"FASTA file not found: {path}")
                fh = open(path, "rb")
                file_handles.append(fh)
                files_payload.append(("files", (path.name, fh, "application/octet-stream")))

            data = {}
            if run_name is not None:
                data["run_name"] = run_name

            return self.post(species, files=files_payload, data=data)
        finally:
            for fh in file_handles:
                fh.close()


CUSTOM_ALIASES: dict[str, str] = {
    "UK": "GBR",
    "GREAT BRITAIN": "GBR",
    "ENGLAND": "GBR",
    "SCOTLAND": "GBR",
    "WALES": "GBR",
    "UNITED KINGDOM": "GBR",
    "RUSSIA": "RUS",
    "RUSSIAN FEDERATION": "RUS",
    "IVORY COAST": "CIV",
    "COTE D'IVOIRE": "CIV",
    "CÔTE D'IVOIRE": "CIV",
    "TURKEY": "TUR",
    "TÜRKIYE": "TUR",
    "SWAZILAND": "SWZ",
    "ESWATINI": "SWZ",
    "USA": "USA",
    "US": "USA",
    "UNITED STATES": "USA",
    "UNITED STATES OF AMERICA": "USA",
    "KOSOVO": "XKX",
    "SOUTH KOREA": "KOR",
    "KOREA, SOUTH": "KOR",
    "NORTH KOREA": "PRK",
    "KOREA, NORTH": "PRK",
    "VIETNAM": "VNM",
    "VIET NAM": "VNM",
    "LAOS": "LAO",
    "SYRIA": "SYR",
    "IRAN": "IRN",
    "MOLDOVA": "MDA",
    "CZECH REPUBLIC": "CZE",
    "CZECHIA": "CZE",
}


def resolve_iso3(country: str) -> str | None:
    """Resolve an arbitrary country name or ISO code to a strict ISO3 format.

    Args:
        country: The raw country name or code to resolve.

    Returns:
        The 3-letter uppercase ISO3 code, or None if unresolvable.
    """
    if not country or not isinstance(country, str):
        return None
    clean = country.strip()
    if not clean:
        return None

    upper_name = clean.upper()
    if upper_name in CUSTOM_ALIASES:
        return CUSTOM_ALIASES[upper_name]
    if upper_name in set(CUSTOM_ALIASES.values()):
        return upper_name

    if len(clean) == 3:
        obj = pycountry.countries.get(alpha_3=upper_name)
        if obj:
            return obj.alpha_3

    if len(clean) == 2:
        obj = pycountry.countries.get(alpha_2=upper_name)
        if obj:
            return obj.alpha_3

    try:
        match = pycountry.countries.lookup(clean)
        return match.alpha_3
    except (LookupError, AttributeError):
        pass

    if len(clean) > 3:
        try:
            results = pycountry.countries.search_fuzzy(clean)
            if results:
                return results[0].alpha_3
        except LookupError:
            pass

    return None


_DEFAULT_CACHE_DIR = Path.home() / ".seroepi" / "cache"


def _raw_fetch_wb_indicator(indicator: str, iso3_tuple: tuple[str, ...], base_url: str) -> list[dict[str, Any]]:
    """Fetch raw indicator records from World Bank API for a tuple of ISO3 codes.

    Args:
        indicator: World Bank indicator ID (e.g. 'SP.DYN.LE00.IN').
        iso3_tuple: Tuple of ISO3 country codes.
        base_url: World Bank API base URL.

    Returns:
        List of raw record dicts returned by the World Bank API.
    """
    if not iso3_tuple:
        return []

    iso3_param = ";".join(sorted(iso3_tuple))
    endpoint_url = f"{base_url.rstrip('/')}/country/{iso3_param}/indicator/{indicator}"

    records: list[dict[str, Any]] = []
    page = 1

    while True:
        resp = requests.get(endpoint_url, params={"format": "json", "per_page": 1000, "page": page}, timeout=15)
        resp.raise_for_status()
        payload = resp.json()

        if not isinstance(payload, list) or len(payload) < 2 or "message" in payload[0]:
            break

        meta, page_records = payload[0], payload[1]
        if not page_records:
            break
        if isinstance(page_records, list):
            records.extend(page_records)

        total_pages = meta.get("pages", 1)
        if page >= total_pages:
            break
        page += 1

    return records


class WorldBankClient:
    """Client for fetching and disk-caching World Bank development indicators."""

    def __init__(self, cache_dir: Path | str | None = None, base_url: str = "https://api.worldbank.org/v2") -> None:
        """Initialize WorldBankClient.

        Args:
            cache_dir: Optional custom directory path for persistent disk caching.
                Defaults to ~/.seroepi/cache.
            base_url: Base URL for the World Bank API.
        """
        self.base_url = base_url
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.memory = joblib.Memory(location=self.cache_dir, verbose=0)
            self._cached_fetch = self.memory.cache(_raw_fetch_wb_indicator)
        except (PermissionError, OSError):
            self.memory = joblib.Memory(location=None, verbose=0)
            self._cached_fetch = self.memory.cache(_raw_fetch_wb_indicator)

    @staticmethod
    def resolve_iso3(country: str) -> str | None:
        """Resolve an arbitrary country name or code to a 3-letter ISO3 code.

        Args:
            country: The raw country string to resolve.

        Returns:
            The 3-letter ISO3 code, or None if unresolvable.
        """
        return resolve_iso3(country)

    def fetch_indicator(self, indicator: str, iso3_codes: list[str]) -> dict[str, dict[int, float]]:
        """Query World Bank API for indicator values across given ISO3 codes.

        Args:
            indicator: World Bank indicator ID (e.g. 'SP.DYN.LE00.IN').
            iso3_codes: List of country names or ISO codes.

        Returns:
            Dict mapping ISO3 code -> {year: float_indicator_value}.
        """
        valid_iso3 = sorted({code for c in iso3_codes if (code := self.resolve_iso3(c))})
        if not valid_iso3:
            return {}

        try:
            records = self._cached_fetch(indicator, tuple(valid_iso3), self.base_url)
        except Exception:
            return {}

        res: dict[str, dict[int, float]] = {}
        for rec in records:
            if not isinstance(rec, dict):
                continue
            country_info = rec.get("country") if isinstance(rec.get("country"), dict) else {}
            raw_iso3 = rec.get("countryiso3code") or country_info.get("id")
            if not raw_iso3:
                continue
            if not isinstance(raw_iso3, str):
                raw_iso3 = str(raw_iso3)
            iso3 = self.resolve_iso3(raw_iso3) or raw_iso3.upper()
            date_str = rec.get("date")
            val = rec.get("value")

            if iso3 and date_str and str(date_str).isdigit() and val is not None:
                try:
                    year = int(date_str)
                    float_val = float(val)
                    if iso3 not in res:
                        res[iso3] = {}
                    res[iso3][year] = float_val
                except (ValueError, TypeError):
                    continue

        return res


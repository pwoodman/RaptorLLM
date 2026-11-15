import { useId, useState, type FormEvent } from "react";
import { Download, Search } from "lucide-react";
import { api, type HfModelSummary, type RepoFile, type SearchSort } from "../../api";
import { useAsyncResource } from "../../hooks/useAsyncResource";
import { useStore } from "../../hooks/useStore";
import {
  budgetFromHardware,
  downloadBlocker,
  rateCompat,
  recommendFile,
  type CompatRating,
  type HardwareBudget,
} from "../../lib/compat";
import { describeError, messageOf } from "../../lib/errors";
import { formatBytes, formatCount } from "../../lib/format";
import { startDownload } from "../../state/downloads";
import { libraryStore } from "../../state/library";
import { EmptyState, ErrorState, LoadingState } from "../AsyncState";
import { Notice } from "../Notice";

const SORT_LABELS: Record<SearchSort, string> = {
  downloads: "Most downloaded",
  likes: "Most liked",
  last_modified: "Recently updated",
};

const RATING_TONE: Record<CompatRating, string> = {
  Great: "tone-ok",
  Ok: "tone-ok",
  Tight: "tone-warn",
  WontFit: "tone-danger",
};

interface Submitted {
  query: string;
  sort: SearchSort;
}

export function DiscoverTab({ onQueued }: { onQueued: () => void }) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<SearchSort>("downloads");
  const [submitted, setSubmitted] = useState<Submitted | null>(null);
  const [repo, setRepo] = useState<string | null>(null);
  const searchId = useId();
  const sortId = useId();

  const results = useAsyncResource<HfModelSummary[]>(
    () => (submitted ? api.models.search(submitted.query, submitted.sort, 30) : Promise.resolve([])),
    [submitted],
  );
  const hardware = useAsyncResource(() => api.hardware.info(), []);

  function submit(event: FormEvent) {
    event.preventDefault();
    const trimmed = query.trim();
    if (trimmed === "") return;
    setRepo(null);
    setSubmitted({ query: trimmed, sort });
  }

  return (
    <div className="discover">
      <form className="search-row" role="search" onSubmit={submit}>
        <div className="field grow">
          <label htmlFor={searchId} className="field-label">
            Search Hugging Face GGUF models
          </label>
          <input
            id={searchId}
            type="search"
            value={query}
            placeholder="For example: qwen coder, llama, gemma"
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor={sortId} className="field-label">
            Sort
          </label>
          <select id={sortId} value={sort} onChange={(e) => setSort(e.target.value as SearchSort)}>
            {(Object.keys(SORT_LABELS) as SearchSort[]).map((value) => (
              <option key={value} value={value}>
                {SORT_LABELS[value]}
              </option>
            ))}
          </select>
        </div>
        <button type="submit" className="btn btn-primary search-submit" disabled={query.trim() === ""}>
          <Search size={15} aria-hidden="true" /> Search
        </button>
      </form>

      {submitted === null && (
        <EmptyState icon={Search} title="Search for a model">
          Results come from Hugging Face. Nothing is downloaded until you choose a file.
        </EmptyState>
      )}
      {submitted !== null && results.loading && <LoadingState label="Searching Hugging Face..." />}
      {submitted !== null && !results.loading && results.error !== null && (
        <ErrorState error={results.error} onRetry={results.reload} />
      )}
      {submitted !== null && !results.loading && results.error === null && results.data?.length === 0 && (
        <EmptyState icon={Search} title="No GGUF repositories found">
          Try a shorter or different name.
        </EmptyState>
      )}

      {results.data && results.data.length > 0 && (
        <div className="discover-grid">
          <ul className="repo-list" aria-label="Search results">
            {results.data.map((model) => (
              <li key={model.id}>
                <button
                  type="button"
                  className="repo-item"
                  aria-pressed={repo === model.id}
                  onClick={() => setRepo(model.id)}
                >
                  <span className="repo-name">{model.id}</span>
                  <span className="muted small">
                    {formatCount(model.downloads)} downloads · {formatCount(model.likes)} likes
                    {model.pipeline_tag ? ` · ${model.pipeline_tag}` : ""}
                  </span>
                </button>
              </li>
            ))}
          </ul>

          <div className="repo-detail">
            {repo === null ? (
              <p className="muted">Select a repository to see its GGUF files.</p>
            ) : (
              <RepoDetail
                repo={repo}
                budget={hardware.data ? budgetFromHardware(hardware.data) : null}
                hardwareError={hardware.error}
                onQueued={onQueued}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function RepoDetail({
  repo,
  budget,
  hardwareError,
  onQueued,
}: {
  repo: string;
  budget: HardwareBudget | null;
  hardwareError: unknown;
  onQueued: () => void;
}) {
  const info = useAsyncResource(() => api.models.repoInfo(repo), [repo]);
  const files = useAsyncResource(() => api.models.repoFiles(repo), [repo]);
  const library = useStore(libraryStore, (s) => s.models);
  const installed = new Set((library ?? []).map((m) => m.id));
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [starting, setStarting] = useState<string | null>(null);

  async function download(file: RepoFile) {
    setError(null);
    setMessage(null);
    setStarting(file.file_name);
    try {
      await startDownload(repo, file.file_name);
      setMessage(`${file.file_name} added to the download queue.`);
    } catch (e) {
      setError(e);
    } finally {
      setStarting(null);
    }
  }

  if (info.loading || files.loading) return <LoadingState label="Reading repository..." />;
  if (info.error !== null) return <ErrorState error={info.error} onRetry={info.reload} />;
  if (files.error !== null) return <ErrorState error={files.error} onRetry={files.reload} />;
  if (!info.data || !files.data) return null;

  const fileList = files.data;
  const recommended = budget ? recommendFile(fileList, budget) : null;

  return (
    <div className="repo-files">
      <h3 className="repo-title">{repo}</h3>
      <p className="muted small">
        {[info.data.license && `License: ${info.data.license}`, info.data.library_name, info.data.pipeline_tag]
          .filter(Boolean)
          .join(" · ") || "No license or library metadata listed."}
      </p>
      {info.data.gated && (
        <Notice tone="warn" title="Gated repository">
          Hugging Face may refuse this download without access granted to your account.
        </Notice>
      )}
      {budget === null && hardwareError !== null && (
        <Notice tone="info" title="Fit ratings unavailable">
          {`Hardware detection failed: ${messageOf(hardwareError)}`}
        </Notice>
      )}
      {recommended && (
        <Notice tone="success" title="Recommended for this machine">
          {recommended.file_name} ({formatBytes(recommended.size)}
          {recommended.quant_label ? `, ${recommended.quant_label}` : ""}). It is the largest file rated Great or Ok.
        </Notice>
      )}
      {fileList.length === 0 && <p className="muted">No top-level GGUF files were found in this repository.</p>}

      {fileList.length > 0 && (
        <div className="table-wrap">
          <table className="data-table">
            <caption className="sr-only">GGUF files in {repo}</caption>
            <thead>
              <tr>
                <th scope="col">File</th>
                <th scope="col">Quant</th>
                <th scope="col">Size</th>
                <th scope="col">Fit</th>
                <th scope="col">
                  <span className="sr-only">Action</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {fileList.map((file) => {
                const id = `${repo}/${file.file_name}`;
                const blocker = downloadBlocker(file);
                const isInstalled = installed.has(id);
                const compat = budget ? rateCompat(file.size, budget) : null;
                return (
                  <tr key={file.file_name}>
                    <td className="break">
                      {file.file_name}
                      {file.split_shard && <span className="tag">split shard</span>}
                    </td>
                    <td>{file.quant_label ?? "Unknown"}</td>
                    <td>{formatBytes(file.size)}</td>
                    <td>
                      {compat ? (
                        <span className={`badge ${RATING_TONE[compat.rating]}`} title={compat.reason}>
                          {compat.rating}
                        </span>
                      ) : (
                        <span className="muted small">Unknown</span>
                      )}
                    </td>
                    <td className="action-cell">
                      {isInstalled ? (
                        <span className="tag">Installed</span>
                      ) : (
                        <button
                          type="button"
                          className="btn btn-sm"
                          disabled={blocker !== null || starting !== null}
                          title={blocker ?? `Download ${file.file_name}`}
                          onClick={() => void download(file)}
                        >
                          <Download size={14} aria-hidden="true" /> Download
                        </button>
                      )}
                      {blocker && <span className="muted small block">{blocker}</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {message !== null && (
        <Notice
          tone="success"
          actions={
            <button type="button" className="btn btn-sm" onClick={onQueued}>
              Open downloads
            </button>
          }
        >
          {message}
        </Notice>
      )}
      {error !== null && (
        <Notice tone="danger" title={describeError(error).title}>
          {messageOf(error)}
        </Notice>
      )}
    </div>
  );
}

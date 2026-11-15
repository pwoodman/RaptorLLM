import { useEffect, useId, useState, type FormEvent } from "react";
import { Pencil, Play, Plus, Star, Trash2 } from "lucide-react";
import {
  presetsDelete,
  presetsList,
  presetsSave,
  presetsSetDefault,
  type Preset,
  type PresetList,
} from "../../api";
import { useStore } from "../../hooks/useStore";
import { messageOf } from "../../lib/errors";
import {
  emptyPresetForm,
  formFromPreset,
  presetFromForm,
  presetSummary,
  PRESET_LIMITS,
  validatePresetForm,
  type PresetErrors,
  type PresetField,
  type PresetForm,
} from "../../lib/presets";
import { libraryStore } from "../../state/library";
import { engineStore, loadModel } from "../../state/engine";
import { ConfirmDialog } from "../ConfirmDialog";
import { EmptyState, ErrorState, LoadingState } from "../AsyncState";
import { Notice } from "../Notice";
import { LoadFailureCard } from "./LoadFailureCard";

type Editor = { mode: "new" } | { mode: "edit"; original: string } | null;

/**
 * Presets of one installed model: named sampling and context settings. A preset can be made the
 * default, applied when loading the model here, and chosen in the chat settings. Ranges are checked
 * here first and again by the backend on every save.
 */
export function PresetsPanel() {
  const models = useStore(libraryStore, (s) => s.models);
  const busy = useStore(engineStore, (s) => s.busy);
  const [modelId, setModelId] = useState("");
  const [list, setList] = useState<PresetList | null>(null);
  const [listError, setListError] = useState<unknown>(null);
  const [editor, setEditor] = useState<Editor>(null);
  const [form, setForm] = useState<PresetForm>(emptyPresetForm);
  const [errors, setErrors] = useState<PresetErrors>({});
  const [saveError, setSaveError] = useState<unknown>(null);
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [actionError, setActionError] = useState<unknown>(null);
  const [deleteName, setDeleteName] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [loadingPreset, setLoadingPreset] = useState<string | null>(null);
  const selectId = useId();

  // Default to the first installed model, and keep the choice valid when the library changes.
  useEffect(() => {
    if (models === null || models.length === 0) return;
    if (!models.some((m) => m.id === modelId)) setModelId(models[0]?.id ?? "");
  }, [models, modelId]);

  useEffect(() => {
    setList(null);
    setListError(null);
    setEditor(null);
    if (modelId === "") return;
    let current = true;
    presetsList(modelId).then(
      (next) => {
        if (current) setList(next);
      },
      (e: unknown) => {
        if (current) setListError(e);
      },
    );
    return () => {
      current = false;
    };
  }, [modelId]);

  function openNew() {
    setEditor({ mode: "new" });
    setForm(emptyPresetForm());
    setErrors({});
    setSaveError(null);
    setNotice(null);
  }

  function openEdit(preset: Preset) {
    setEditor({ mode: "edit", original: preset.name });
    setForm(formFromPreset(preset));
    setErrors({});
    setSaveError(null);
    setNotice(null);
  }

  function setField(field: PresetField, value: string) {
    setForm((current) => ({ ...current, [field]: value }));
    setErrors((current) => ({ ...current, [field]: undefined }));
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    const problems = validatePresetForm(form);
    setErrors(problems);
    setSaveError(null);
    if (Object.keys(problems).length > 0) return;
    setSaving(true);
    try {
      const saved = await presetsSave(presetFromForm(form, modelId));
      // Renaming an edited preset replaces the old name, so the old one is removed.
      if (editor?.mode === "edit" && editor.original !== saved.name) {
        await presetsDelete(modelId, editor.original);
      }
      setList(await presetsList(modelId));
      setEditor(null);
      setNotice(`Saved preset "${saved.name}".`);
    } catch (e) {
      setSaveError(e);
    } finally {
      setSaving(false);
    }
  }

  async function run(action: () => Promise<PresetList>, done: string) {
    setActionError(null);
    setNotice(null);
    try {
      setList(await action());
      setNotice(done);
    } catch (e) {
      setActionError(e);
    }
  }

  async function confirmDelete() {
    if (deleteName === null) return;
    setDeleting(true);
    await run(() => presetsDelete(modelId, deleteName), `Deleted preset "${deleteName}".`);
    setDeleting(false);
    setDeleteName(null);
  }

  async function loadWith(name: string) {
    setLoadingPreset(name);
    setActionError(null);
    setNotice(null);
    try {
      await loadModel(modelId, name);
      setNotice(`Loaded with preset "${name}".`);
    } catch (e) {
      setActionError(e);
    } finally {
      setLoadingPreset(null);
    }
  }

  if (models === null) return <LoadingState label="Reading installed models..." />;
  if (models.length === 0) {
    return (
      <EmptyState icon={Pencil} title="No presets yet">
        Install a model first. Presets belong to an installed model.
      </EmptyState>
    );
  }

  const presets = list?.presets ?? [];

  return (
    <section className="presets" aria-labelledby={`${selectId}-heading`}>
      <h2 id={`${selectId}-heading`} className="subsection-title">
        Presets
      </h2>
      <p className="muted small">
        A preset holds a system prompt, context length, sampling values and a reply limit for one model. Use "Load
        with this preset" to apply one when loading. A default is marked here; it is not applied automatically.
      </p>

      <div className="field">
        <label htmlFor={selectId} className="field-label">
          Model
        </label>
        <select id={selectId} value={modelId} onChange={(e) => setModelId(e.target.value)}>
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.file_name}
            </option>
          ))}
        </select>
      </div>

      {listError !== null && <ErrorState error={listError} />}
      {listError === null && list === null && <LoadingState label="Reading presets..." />}

      {list !== null && presets.length === 0 && editor === null && (
        <p className="muted small">No presets for this model yet.</p>
      )}

      {presets.length > 0 && (
        <ul className="preset-list" aria-label="Presets of this model">
          {presets.map((preset) => {
            const isDefault = list?.default === preset.name;
            return (
              <li key={preset.name} className="preset-row">
                <div className="preset-row-main">
                  <strong className="break">{preset.name}</strong>
                  {isDefault && <span className="badge tone-ok">Default</span>}
                  <span className="muted small">{presetSummary(preset)}</span>
                  {preset.system_prompt && <span className="muted small">System prompt set</span>}
                </div>
                <div className="button-row">
                  <button
                    type="button"
                    className="btn btn-sm btn-primary"
                    disabled={busy !== null || loadingPreset !== null}
                    onClick={() => void loadWith(preset.name)}
                  >
                    <Play size={13} aria-hidden="true" />{" "}
                    {loadingPreset === preset.name ? "Loading..." : "Load with this preset"}
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm"
                    onClick={() => void run(() => presetsSetDefault(modelId, isDefault ? null : preset.name), isDefault ? "Default cleared." : `"${preset.name}" is now the default.`)}
                  >
                    <Star size={13} aria-hidden="true" /> {isDefault ? "Clear default" : "Make default"}
                  </button>
                  <button type="button" className="btn btn-sm" onClick={() => openEdit(preset)} aria-label={`Edit ${preset.name}`}>
                    <Pencil size={13} aria-hidden="true" /> Edit
                  </button>
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm icon-text-danger"
                    onClick={() => setDeleteName(preset.name)}
                    aria-label={`Delete ${preset.name}`}
                  >
                    <Trash2 size={13} aria-hidden="true" /> Delete
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}

      {editor === null && (
        <button type="button" className="btn btn-sm" onClick={openNew} disabled={list === null}>
          <Plus size={13} aria-hidden="true" /> New preset
        </button>
      )}

      {editor !== null && (
        <PresetEditor
          mode={editor.mode}
          form={form}
          errors={errors}
          saving={saving}
          onField={setField}
          onSubmit={(e) => void save(e)}
          onCancel={() => setEditor(null)}
        />
      )}

      {saveError !== null && (
        <Notice tone="danger" title="Preset not saved">
          {messageOf(saveError)}
        </Notice>
      )}
      {notice !== null && <Notice tone="success">{notice}</Notice>}
      {actionError !== null && <LoadFailureCard error={actionError} />}

      <ConfirmDialog
        open={deleteName !== null}
        title="Delete this preset?"
        confirmLabel="Delete preset"
        tone="danger"
        busy={deleting}
        onCancel={() => setDeleteName(null)}
        onConfirm={() => void confirmDelete()}
      >
        <p>
          The preset "{deleteName}" is removed for this model. The model file is not touched. This cannot be undone.
        </p>
      </ConfirmDialog>
    </section>
  );
}

interface PresetEditorProps {
  mode: "new" | "edit";
  form: PresetForm;
  errors: PresetErrors;
  saving: boolean;
  onField: (field: PresetField, value: string) => void;
  onSubmit: (event: FormEvent) => void;
  onCancel: () => void;
}

function PresetEditor({ mode, form, errors, saving, onField, onSubmit, onCancel }: PresetEditorProps) {
  const id = useId();
  const fieldId = (name: string) => `${id}-${name}`;
  const hint = (name: PresetField) =>
    errors[name] ? { "aria-invalid": true as const, "aria-describedby": `${fieldId(name)}-error` } : {};

  return (
    <form className="preset-editor" onSubmit={onSubmit} aria-label={mode === "new" ? "New preset" : "Edit preset"} noValidate>
      <h3 className="subsection-title">{mode === "new" ? "New preset" : "Edit preset"}</h3>
      <div className="params-grid">
        <Text id={fieldId("name")} label="Name" value={form.name} error={errors.name} onChange={(v) => onField("name", v)} maxLength={PRESET_LIMITS.nameChars} aria={hint("name")} />
        <Num id={fieldId("nCtx")} label="Context length (tokens)" value={form.nCtx} error={errors.nCtx} onChange={(v) => onField("nCtx", v)} aria={hint("nCtx")} />
        <Num id={fieldId("temperature")} label="Temperature" value={form.temperature} error={errors.temperature} onChange={(v) => onField("temperature", v)} aria={hint("temperature")} />
        <Num id={fieldId("topP")} label="Top-p" value={form.topP} error={errors.topP} onChange={(v) => onField("topP", v)} aria={hint("topP")} />
        <Num id={fieldId("topK")} label="Top-k (0 turns it off)" value={form.topK} error={errors.topK} onChange={(v) => onField("topK", v)} aria={hint("topK")} />
        <Num id={fieldId("repeatPenalty")} label="Repeat penalty" value={form.repeatPenalty} error={errors.repeatPenalty} onChange={(v) => onField("repeatPenalty", v)} aria={hint("repeatPenalty")} />
        <Num id={fieldId("maxTokens")} label="Reply limit (blank for none)" value={form.maxTokens} error={errors.maxTokens} onChange={(v) => onField("maxTokens", v)} aria={hint("maxTokens")} />
      </div>
      <div className="field">
        <label htmlFor={fieldId("systemPrompt")} className="field-label">
          System prompt <span className="muted">(optional)</span>
        </label>
        <textarea
          id={fieldId("systemPrompt")}
          rows={3}
          value={form.systemPrompt}
          onChange={(e) => onField("systemPrompt", e.target.value)}
          {...hint("systemPrompt")}
        />
        {errors.systemPrompt && (
          <span id={`${fieldId("systemPrompt")}-error`} className="warn-text small">
            {errors.systemPrompt}
          </span>
        )}
      </div>
      <div className="button-row">
        <button type="submit" className="btn btn-primary" disabled={saving}>
          {saving ? "Saving..." : "Save preset"}
        </button>
        <button type="button" className="btn btn-ghost" onClick={onCancel} disabled={saving}>
          Cancel
        </button>
      </div>
    </form>
  );
}

function Text(props: {
  id: string;
  label: string;
  value: string;
  error?: string;
  maxLength?: number;
  onChange: (value: string) => void;
  aria: Record<string, unknown>;
}) {
  return (
    <div className="field">
      <label htmlFor={props.id} className="field-label">
        {props.label}
      </label>
      <input id={props.id} value={props.value} maxLength={props.maxLength} onChange={(e) => props.onChange(e.target.value)} {...props.aria} />
      {props.error && (
        <span id={`${props.id}-error`} className="warn-text small">
          {props.error}
        </span>
      )}
    </div>
  );
}

function Num(props: {
  id: string;
  label: string;
  value: string;
  error?: string;
  onChange: (value: string) => void;
  aria: Record<string, unknown>;
}) {
  return (
    <div className="field">
      <label htmlFor={props.id} className="field-label">
        {props.label}
      </label>
      <input id={props.id} type="text" inputMode="decimal" value={props.value} onChange={(e) => props.onChange(e.target.value)} {...props.aria} />
      {props.error && (
        <span id={`${props.id}-error`} className="warn-text small">
          {props.error}
        </span>
      )}
    </div>
  );
}

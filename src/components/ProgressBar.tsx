interface ProgressBarProps {
  label: string;
  value: number;
  max: number;
}

/** Progress bar. A zero or unknown total shows an indeterminate state instead of a false 0%. */
export function ProgressBar({ label, value, max }: ProgressBarProps) {
  if (max <= 0) {
    return <div className="progress progress-indeterminate" role="progressbar" aria-label={label} />;
  }
  const percent = Math.min(100, Math.max(0, Math.round((value / max) * 100)));
  return (
    <div
      className="progress"
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={percent}
    >
      <div className="progress-fill" style={{ width: `${percent}%` }} />
    </div>
  );
}

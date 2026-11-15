import { useId, type ReactNode } from "react";

interface PanelProps {
  id?: string;
  title: string;
  description?: ReactNode;
  icon?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}

export function Panel({ id, title, description, icon, actions, children }: PanelProps) {
  const headingId = useId();
  return (
    <section id={id} className="panel" aria-labelledby={headingId}>
      <header className="panel-head">
        <div className="panel-heading">
          {icon && <span className="panel-icon">{icon}</span>}
          <div>
            <h2 id={headingId}>{title}</h2>
            {description && <p className="muted">{description}</p>}
          </div>
        </div>
        {actions && <div className="panel-actions">{actions}</div>}
      </header>
      <div className="panel-body">{children}</div>
    </section>
  );
}

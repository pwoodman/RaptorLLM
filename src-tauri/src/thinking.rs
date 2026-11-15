//! Thinking effort as the app shows it: what the loaded model supports and which of the template's
//! levels each of Low, Medium, High and Max means for it. The mapping itself lives in
//! `raptor_gateway::thinking`; the gateway and the in-app chat both use it, so a request means the
//! same thing from either door.

use raptor_gateway::{level_map, map_effort, Effort, ThinkingCaps};
use serde::Serialize;

/// One app level and the template level it maps to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EffortChoice {
    pub effort: Effort,
    pub model_level: String,
}

/// What the chat window needs to show the effort control.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ThinkingView {
    /// False for a model with no effort control: the window shows nothing and sends nothing.
    pub supported: bool,
    /// The template's own level names.
    pub model_levels: Vec<String>,
    /// The four app levels with the template level each maps to (empty when unsupported). Two
    /// entries with the same `model_level` are one setting on this model.
    pub choices: Vec<EffortChoice>,
    /// The level used when a chat does not choose (the setting).
    pub default_effort: Effort,
}

impl ThinkingView {
    pub fn from_caps(caps: &ThinkingCaps, default_effort: Effort) -> Self {
        Self {
            supported: caps.supported,
            model_levels: caps.levels.clone(),
            choices: level_map(caps)
                .into_iter()
                .map(|(effort, model_level)| EffortChoice {
                    effort,
                    model_level,
                })
                .collect(),
            default_effort,
        }
    }
}

/// The template level to send with one chat request, or `None` when the model has no effort
/// control (then the request carries no reasoning field at all). `chosen` is the conversation's
/// own selection; without one the setting applies.
pub fn level_for_request(
    caps: &ThinkingCaps,
    chosen: Option<Effort>,
    default_effort: Effort,
) -> Option<String> {
    map_effort(chosen.unwrap_or(default_effort), caps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen() -> ThinkingCaps {
        ThinkingCaps {
            supported: true,
            levels: vec!["low".into(), "medium".into(), "xhigh".into()],
        }
    }

    #[test]
    fn the_view_for_qwen_lists_four_choices_and_shows_that_high_and_max_share_a_level() {
        let v = ThinkingView::from_caps(&qwen(), Effort::Medium);
        assert!(v.supported);
        assert_eq!(v.model_levels, ["low", "medium", "xhigh"]);
        let pairs: Vec<_> = v
            .choices
            .iter()
            .map(|c| (c.effort, c.model_level.as_str()))
            .collect();
        assert_eq!(
            pairs,
            [
                (Effort::Low, "low"),
                (Effort::Medium, "medium"),
                (Effort::High, "xhigh"),
                (Effort::Max, "xhigh"),
            ]
        );
        assert_eq!(v.default_effort, Effort::Medium);
    }

    #[test]
    fn the_view_for_a_model_without_the_capability_offers_nothing() {
        let v = ThinkingView::from_caps(&ThinkingCaps::default(), Effort::High);
        assert!(!v.supported);
        assert!(v.choices.is_empty() && v.model_levels.is_empty());
        // The setting is still reported, so the Settings page and the chat agree.
        assert_eq!(v.default_effort, Effort::High);
    }

    #[test]
    fn the_view_serializes_with_lowercase_effort_names() {
        let v = ThinkingView::from_caps(&qwen(), Effort::Max);
        let json = serde_json::to_value(&v).unwrap();
        assert_eq!(json["default_effort"], "max");
        assert_eq!(json["choices"][2]["effort"], "high");
        assert_eq!(json["choices"][2]["model_level"], "xhigh");
    }

    #[test]
    fn a_request_uses_the_conversations_choice_else_the_setting_else_nothing() {
        let caps = qwen();
        assert_eq!(
            level_for_request(&caps, Some(Effort::Low), Effort::Medium).as_deref(),
            Some("low")
        );
        assert_eq!(
            level_for_request(&caps, None, Effort::Medium).as_deref(),
            Some("medium")
        );
        assert_eq!(
            level_for_request(&caps, None, Effort::Max).as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            level_for_request(&ThinkingCaps::default(), Some(Effort::High), Effort::Medium),
            None,
            "a non-thinking model gets no reasoning field whatever was chosen"
        );
    }
}

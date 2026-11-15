//! Thinking effort for models that can reason.
//!
//! llama-server reports what a model's chat template supports at `GET /props`, under
//! `chat_template_caps`: `supports_reasoning_effort` and the template's own level names
//! (`reasoning_effort_levels`, Qwen3.8: `low`, `medium`, `xhigh`). A request may carry
//! `reasoning_effort` to pick one without reloading the model. The app offers four levels, Low,
//! Medium, High and Max, whatever the template calls its own; [`map_effort`] translates.
//!
//! A model without `supports_reasoning_effort` is not touched at all: no selector, no field added.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The four levels the app offers. Medium is the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    Low,
    #[default]
    Medium,
    High,
    Max,
}

impl Effort {
    /// Every level, lowest first.
    pub const ALL: [Effort; 4] = [Effort::Low, Effort::Medium, Effort::High, Effort::Max];

    /// The lowercase name used in settings and on the wire.
    pub fn as_str(self) -> &'static str {
        match self {
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::Max => "max",
        }
    }
}

/// What the loaded model's chat template says about reasoning effort.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingCaps {
    /// True when the template accepts a reasoning effort and names at least one level.
    pub supported: bool,
    /// The template's own level names, lowercase, in the order it lists them.
    pub levels: Vec<String>,
}

impl ThinkingCaps {
    /// Reads llama-server's `/props` body. Anything missing or malformed means "not supported": a
    /// model is never offered a control it may not have. A template that accepts an effort but
    /// lists no level names also counts as unsupported, because there is no name to send.
    pub fn from_props(props: &Value) -> Self {
        let caps = &props["chat_template_caps"];
        if caps["supports_reasoning_effort"].as_bool() != Some(true) {
            return Self::default();
        }
        let mut levels: Vec<String> = Vec::new();
        for name in caps["reasoning_effort_levels"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            let name = name.trim().to_ascii_lowercase();
            if !name.is_empty() && !levels.contains(&name) {
                levels.push(name);
            }
        }
        Self {
            supported: !levels.is_empty(),
            levels,
        }
    }
}

/// Order of the level names llama.cpp knows. Names outside this list are never chosen.
fn rank(level: &str) -> Option<u8> {
    Some(match level {
        "minimal" => 0,
        "low" => 1,
        "medium" => 2,
        "high" => 3,
        "xhigh" => 4,
        "max" => 5,
        _ => return None,
    })
}

const MEDIUM_RANK: u8 = 2;

/// The template's own level for one of the app's levels, or `None` when the model has no effort
/// control. Low and Medium use `low` and `medium` when the template has them; High uses `high`,
/// else the highest level at or above medium (`xhigh`), else the highest there is; Max uses `max`,
/// else the highest there is. A missing Low is the lowest level; a missing Medium is the nearest
/// one, the lower of two equally near.
pub fn map_effort(effort: Effort, caps: &ThinkingCaps) -> Option<String> {
    if !caps.supported {
        return None;
    }
    let mut ranked: Vec<(u8, &str)> = caps
        .levels
        .iter()
        .filter_map(|l| rank(l).map(|r| (r, l.as_str())))
        .collect();
    ranked.sort_by_key(|(r, _)| *r);
    let (lowest, highest) = (ranked.first()?.1, ranked.last()?.1);
    let has = |name: &str| ranked.iter().any(|(_, l)| *l == name);
    let pick = match effort {
        Effort::Low if has("low") => "low",
        Effort::Low => lowest,
        Effort::Medium if has("medium") => "medium",
        Effort::Medium => ranked
            .iter()
            .min_by_key(|(r, _)| r.abs_diff(MEDIUM_RANK))
            .map(|(_, l)| *l)?,
        Effort::High if has("high") => "high",
        Effort::High => ranked
            .iter()
            .rev()
            .find(|(r, _)| *r >= MEDIUM_RANK)
            .map(|(_, l)| *l)
            .unwrap_or(highest),
        Effort::Max if has("max") => "max",
        Effort::Max => highest,
    };
    Some(pick.to_string())
}

/// The template's level for each of the four app levels, lowest first. Empty without effort
/// control. Two app levels with the same template level are a shared setting; the UI says so.
pub fn level_map(caps: &ThinkingCaps) -> Vec<(Effort, String)> {
    Effort::ALL
        .into_iter()
        .filter_map(|e| map_effort(e, caps).map(|l| (e, l)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn caps(levels: &[&str]) -> ThinkingCaps {
        ThinkingCaps {
            supported: true,
            levels: levels.iter().map(|l| l.to_string()).collect(),
        }
    }

    fn qwen_props() -> Value {
        // Measured: Qwen3.8-27B, llama-server b12374 --jinja.
        json!({
            "modalities": {"vision": false},
            "chat_template_caps": {
                "supports_tools": true,
                "supports_reasoning_effort": true,
                "reasoning_effort_levels": ["low", "medium", "xhigh"]
            }
        })
    }

    #[test]
    fn effort_defaults_to_medium_and_names_are_lowercase() {
        assert_eq!(Effort::default(), Effort::Medium);
        assert_eq!(
            serde_json::to_string(&Effort::Max).unwrap(),
            "\"max\"",
            "wire format"
        );
        assert_eq!(
            serde_json::from_str::<Effort>("\"high\"").unwrap(),
            Effort::High
        );
        assert!(serde_json::from_str::<Effort>("\"ultra\"").is_err());
        for e in Effort::ALL {
            assert_eq!(
                serde_json::from_str::<Effort>(&format!("\"{}\"", e.as_str())).unwrap(),
                e
            );
        }
    }

    #[test]
    fn qwens_props_are_read_as_a_model_with_three_levels() {
        let c = ThinkingCaps::from_props(&qwen_props());
        assert!(c.supported);
        assert_eq!(c.levels, ["low", "medium", "xhigh"]);
    }

    #[test]
    fn a_model_without_the_capability_is_not_supported() {
        for props in [
            json!({}),
            json!({"chat_template_caps": {}}),
            json!({"chat_template_caps": {"supports_reasoning_effort": false,
                "reasoning_effort_levels": ["low"]}}),
            json!({"chat_template_caps": {"supports_reasoning_effort": "yes"}}),
            // Accepts an effort but names no level: nothing to send.
            json!({"chat_template_caps": {"supports_reasoning_effort": true}}),
            json!({"chat_template_caps": {"supports_reasoning_effort": true,
                "reasoning_effort_levels": []}}),
        ] {
            assert_eq!(ThinkingCaps::from_props(&props), ThinkingCaps::default());
        }
    }

    #[test]
    fn level_names_are_lowercased_deduplicated_and_non_strings_dropped() {
        let c = ThinkingCaps::from_props(&json!({"chat_template_caps": {
            "supports_reasoning_effort": true,
            "reasoning_effort_levels": ["Low", "low", 7, " High ", ""]
        }}));
        assert_eq!(c.levels, ["low", "high"]);
    }

    #[test]
    fn qwens_levels_map_low_medium_xhigh_xhigh() {
        let c = ThinkingCaps::from_props(&qwen_props());
        assert_eq!(map_effort(Effort::Low, &c).as_deref(), Some("low"));
        assert_eq!(map_effort(Effort::Medium, &c).as_deref(), Some("medium"));
        assert_eq!(map_effort(Effort::High, &c).as_deref(), Some("xhigh"));
        assert_eq!(map_effort(Effort::Max, &c).as_deref(), Some("xhigh"));
        let shared: Vec<_> = level_map(&c).into_iter().map(|(_, l)| l).collect();
        assert_eq!(shared, ["low", "medium", "xhigh", "xhigh"]);
    }

    #[test]
    fn a_template_with_all_the_names_maps_one_to_one() {
        let c = caps(&["minimal", "low", "medium", "high", "xhigh", "max"]);
        let got: Vec<_> = Effort::ALL
            .into_iter()
            .map(|e| map_effort(e, &c).unwrap())
            .collect();
        assert_eq!(got, ["low", "medium", "high", "max"]);
    }

    #[test]
    fn missing_levels_fall_back_to_the_nearest_sensible_one() {
        // Only high and max: Low is the lowest, Medium the nearest (high), High exact, Max exact.
        let c = caps(&["high", "max"]);
        assert_eq!(map_effort(Effort::Low, &c).as_deref(), Some("high"));
        assert_eq!(map_effort(Effort::Medium, &c).as_deref(), Some("high"));
        assert_eq!(map_effort(Effort::High, &c).as_deref(), Some("high"));
        assert_eq!(map_effort(Effort::Max, &c).as_deref(), Some("max"));
        // Only low and medium: High is the highest at or above medium, Max the highest.
        let c = caps(&["low", "medium"]);
        assert_eq!(map_effort(Effort::High, &c).as_deref(), Some("medium"));
        assert_eq!(map_effort(Effort::Max, &c).as_deref(), Some("medium"));
        // Only low: nothing is at or above medium, so High and Max use the highest there is.
        let c = caps(&["low"]);
        for e in Effort::ALL {
            assert_eq!(map_effort(e, &c).as_deref(), Some("low"));
        }
        // Low and high: Medium is equally near both; the lower wins.
        let c = caps(&["low", "high"]);
        assert_eq!(map_effort(Effort::Medium, &c).as_deref(), Some("low"));
    }

    #[test]
    fn the_order_the_template_lists_levels_in_does_not_matter() {
        let c = caps(&["xhigh", "low", "medium"]);
        assert_eq!(map_effort(Effort::High, &c).as_deref(), Some("xhigh"));
        assert_eq!(map_effort(Effort::Low, &c).as_deref(), Some("low"));
    }

    #[test]
    fn unknown_level_names_are_never_chosen_and_all_unknown_means_no_control() {
        let c = caps(&["turbo", "low"]);
        for e in Effort::ALL {
            assert_eq!(map_effort(e, &c).as_deref(), Some("low"));
        }
        let only_unknown = caps(&["turbo"]);
        assert_eq!(map_effort(Effort::Medium, &only_unknown), None);
        assert!(level_map(&only_unknown).is_empty());
    }

    #[test]
    fn an_unsupported_model_maps_to_nothing() {
        assert_eq!(map_effort(Effort::Medium, &ThinkingCaps::default()), None);
        assert!(level_map(&ThinkingCaps::default()).is_empty());
    }
}

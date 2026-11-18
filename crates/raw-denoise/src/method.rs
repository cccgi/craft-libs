//! The denoiser choices an app offers, by id, and the runner each makes for a photo.

use serde::{Deserialize, Serialize};

use crate::builtin::{self, Builtin};

/// Which denoiser runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Method {
    /// The built-in denoiser ([`crate::builtin`]): block matching on the variance-stabilised photo, its noise measured
    /// from the photo itself.
    #[default]
    #[serde(rename = "builtin")]
    Builtin,
    /// The built-in denoiser followed by the light non-local-means pass ([`crate::nlm_post`]).
    #[serde(rename = "builtin-nlm")]
    BuiltinNlm,
}

impl Method {
    /// Every choice, the default first.
    pub const ALL: [Method; 2] = [Method::Builtin, Method::BuiltinNlm];

    /// The id apps store and pass around: [`builtin::ID`] or [`builtin::NLM_ID`].
    pub fn id(self) -> &'static str {
        match self {
            Method::Builtin => builtin::ID,
            Method::BuiltinNlm => builtin::NLM_ID,
        }
    }

    /// The choice an id names, if any.
    pub fn from_id(id: &str) -> Option<Method> {
        Method::ALL.into_iter().find(|m| m.id() == id)
    }

    /// The name shown to people.
    pub fn name(self) -> &'static str {
        match self {
            Method::Builtin => "Built-in",
            Method::BuiltinNlm => "Non-local means (after built-in)",
        }
    }

    /// The algorithm's version: it changes when the pictures it makes do, so a cache keyed by it is made again.
    pub fn version(self) -> String {
        match self {
            Method::Builtin => builtin::VERSION.to_string(),
            Method::BuiltinNlm => builtin::nlm_version(),
        }
    }

    /// The tile runner for a photo of noise `noise` (`(a, b)`, variance = `a · signal + b`; `None`: each tile measures
    /// its own), with the default tuning.
    pub fn runner(self, noise: Option<(f32, f32)>) -> Builtin {
        let b = Builtin { noise, ..Default::default() };
        match self {
            Method::Builtin => b,
            Method::BuiltinNlm => b.with_nlm(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ids round-trip, unknown ids are refused, the default is the plain built-in denoiser, and serde uses the ids.
    #[test]
    fn ids_round_trip_and_the_default_is_the_builtin() {
        for m in Method::ALL {
            assert_eq!(Method::from_id(m.id()), Some(m));
            assert_eq!(serde_json::to_string(&m).unwrap(), format!("\"{}\"", m.id()));
            assert_eq!(serde_json::from_str::<Method>(&format!("\"{}\"", m.id())).unwrap(), m);
        }
        assert_eq!(Method::from_id("comfy"), None);
        assert_eq!(Method::from_id(""), None);
        assert_eq!(Method::default(), Method::Builtin);
        assert_ne!(Method::Builtin.version(), Method::BuiltinNlm.version());
        assert!(Method::Builtin.runner(None).post.is_none());
        assert!(Method::BuiltinNlm.runner(Some((0.01, 1e-3))).post.is_some());
    }
}

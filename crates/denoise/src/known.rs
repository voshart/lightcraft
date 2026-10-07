//! The denoise models LightCraft knows about: what each one is, who may use it, and where to fetch it.
//!
//! None is part of LightCraft. A model with an address here can be downloaded when the user asks and has accepted
//! its terms; anything else is brought by the user as an `.onnx` file with a `denoise-model.json` beside it.

use lightcraft_faces::known::Download;
use lightcraft_faces::manifest::{Commercial, Licence};

use crate::manifest::{DenoiserManifest, Domain, Gain};

/// A known model, and how to get it.
pub struct Known {
    pub manifest: DenoiserManifest,
    /// What to fetch (an archive or the `.onnx` itself); `None` when the user must bring the file.
    pub download: Option<Download>,
    /// The `.onnx` inside the downloaded archive (`None`: the download is the `.onnx`).
    pub entry: Option<&'static str>,
}

/// RawNIND's UtNet2 for Bayer sensors, as published for darktable (`rawdenoise-nind.dtmodel`, a zip with a Bayer and a
/// linear model). Its weights are GPL-3.0; the training pictures are CC BY 4.0 and CC0.
fn rawnind_bayer() -> Known {
    let manifest = DenoiserManifest {
        id: "rawnind-bayer".into(),
        name: "RawNIND UtNet2 (Bayer)".into(),
        version: "1.0".into(),
        licence: Licence {
            name: "GPL-3.0".into(),
            commercial: Commercial::Yes,
            url: Some("https://www.gnu.org/licenses/gpl-3.0.html".into()),
            notice: "The weights are published under the GNU General Public License v3. LightCraft does not include them: they \
                     are downloaded to your computer at your request and used by it only. If you pass the model file on, the \
                     GPL applies to that copy."
                .into(),
        },
        source: Some("https://github.com/darktable-org/darktable-ai/tree/master/models/rawdenoise-nind".into()),
        sha256: Some("da27509dab6a2915da67e988acd86cf71f9d5bbc8d1aa0ed32933578a887b901".into()),
        size_bytes: Some(31_056_425),
        provenance: "Trained on RawNIND, real photographs of the same scenes taken noisy and clean (images CC BY 4.0 and CC0, \
                     published on Wikimedia Commons and by the University of Louvain). Paper: arXiv 2501.08924."
            .into(),
        domain: Domain::BayerToRgb,
        tile: 512,
        overlap: 64,
        gain: Gain::MatchMean { nominal: 1.0e6, max_deviation: 0.05 },
    };
    let download = Download {
        id: manifest.id.clone(),
        url: "https://github.com/darktable-org/darktable-ai/releases/download/release-5.6.0/rawdenoise-nind.dtmodel".into(),
        file_name: "rawdenoise-nind.dtmodel".into(),
        size_bytes: 57_700_134,
        sha256: "d71b5f1e727c85a359e6f74dca9e2016c9d8fc3e2f7ac3e9b347d80ceca969af".into(),
    };
    Known { manifest, download: Some(download), entry: Some("model_bayer.onnx") }
}

/// Every model LightCraft knows.
pub fn all() -> Vec<Known> {
    vec![rawnind_bayer()]
}

/// The known model with this id.
pub fn find(id: &str) -> Option<Known> {
    all().into_iter().find(|k| k.manifest.id == id)
}

/// The known model whose `.onnx` has this SHA-256.
pub fn by_sha256(sha256: &str) -> Option<DenoiserManifest> {
    all().into_iter().map(|k| k.manifest).find(|m| m.sha256.as_deref() == Some(sha256))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::validate;

    #[test]
    fn known_models_are_valid_unique_and_downloaded_over_https() {
        let all = all();
        assert!(!all.is_empty());
        for (i, k) in all.iter().enumerate() {
            validate(&k.manifest).unwrap();
            assert!(all.iter().skip(i + 1).all(|o| o.manifest.id != k.manifest.id), "unique ids");
            assert!(!k.manifest.licence.name.is_empty() && !k.manifest.licence.notice.is_empty(), "terms are shown before use");
            if let Some(d) = &k.download {
                assert_eq!(d.id, k.manifest.id);
                assert!(d.url.starts_with("https://") && d.sha256.len() == 64 && d.size_bytes > 0);
                // a name that cannot leave the staging folder
                assert!(!d.file_name.contains(['/', '\\']) && !d.file_name.starts_with('.'));
            }
            // a model that may not be used commercially is never offered for download
            if k.manifest.licence.commercial == Commercial::No {
                assert!(k.download.is_none(), "{}", k.manifest.id);
            }
        }
        assert!(find("rawnind-bayer").is_some() && find("nothing").is_none());
        assert_eq!(by_sha256("da27509dab6a2915da67e988acd86cf71f9d5bbc8d1aa0ed32933578a887b901").map(|m| m.id), Some("rawnind-bayer".into()));
    }
}

use std::collections::HashMap;

use crate::connect::YncPluginClient;

use super::super::request::TranslationRequest;

pub(in crate::translation) fn translate_text(
    request: &TranslationRequest,
) -> anyhow::Result<Vec<(String, String)>> {
    let mut client = YncPluginClient::for_command(request.config.translation_plugin_http_port)?;
    if request.targets.len() == 1 {
        let target = &request.targets[0];
        let response =
            client.translate(&request.source_recognition_id, target, &request.source_text)?;
        return Ok(vec![(target.clone(), response.text)]);
    }

    let response = client.translates(
        &request.source_recognition_id,
        &request.targets,
        &request.source_text,
    )?;
    let translations_by_lang = response
        .result
        .into_iter()
        .map(|result| (result.lang, result.text))
        .collect::<HashMap<_, _>>();
    Ok(request
        .targets
        .iter()
        .filter_map(|target| {
            translations_by_lang
                .get(target)
                .map(|text| (target.clone(), text.clone()))
        })
        .collect())
}

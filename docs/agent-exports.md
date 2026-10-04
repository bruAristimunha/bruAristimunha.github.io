# Public agent exports

Jekyll builds `/llms.txt`, `/bio.md`, `/publications.json` and `/publications.bib`.
No plugin, remote API, or separate generated-data commit is required.

- Edit publication data only in `_data/publications.yml`. JSON preserves its array
  and all fields unchanged (including absent statuses); BibTeX concatenates the
  existing `bibtex` strings. Relative asset URLs resolve against the site root.
- Edit biography and research interests in `_includes/bio.md` and
  `_includes/research-interests.md`. Both the homepage and Markdown export include
  these same texts; do not maintain an independent biography in the export.
- The JSON is the site's curated publication list, not a claim of completeness.
  An absent status is unspecified; submitted work is not accepted work.
- `layout: null` prevents HTML wrappers. The Markdown endpoint uses a `.txt`
  source with `/bio.md` permalink to bypass Markdown-to-HTML conversion.

Verification (also run before the Pages artifact upload):

```sh
JEKYLL_ENV=production bundle exec jekyll build
bundle exec ruby docs/check-agent-exports.rb _site
```

The check compares built JSON with canonical YAML, visible IDs/titles/venues/
statuses and citation attributes, exact BibTeX text and unique keys, local
publication assets, directory URLs and fragments, shared biography, and absence
of template/layout/private metadata leakage. External scholarly sites are not
fetched: these exports preserve the existing links without asserting availability.

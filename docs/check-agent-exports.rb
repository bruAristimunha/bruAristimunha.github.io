#!/usr/bin/env ruby
# Run after `bundle exec jekyll build`: bundle exec ruby docs/check-agent-exports.rb _site
require 'json'
require 'yaml'
require 'nokogiri'
require 'uri'

root = File.expand_path('..', __dir__)
site = File.expand_path(ARGV.fetch(0, '_site'))
def check(condition, message)
  abort "FAIL: #{message}" unless condition
end
pubs = YAML.safe_load(File.read(File.join(root, '_data/publications.yml')))
export = JSON.parse(File.read(File.join(site, 'publications.json')))
check(export == pubs, 'JSON must preserve every canonical field, including missing statuses')
html = Nokogiri::HTML(File.read(File.join(site, 'index.html')))
entries = html.css('article.pubtrack__entry')
check(entries.size == pubs.size, 'visible and exported publication counts')
pubs.each do |pub|
  entry = entries.find { |e| e['id'] == pub['id'] }
  check(entry, "visible ID #{pub['id']}")
  check(entry.at_css('.pubtrack__title').text == pub['title'], "title #{pub['id']}")
  check(entry.at_css('.pubtrack__venue').text == pub['venue'], "venue #{pub['id']}")
  check(entry.at_css('.pubtrack__status')&.text == pub['status']&.tr('-', ' '), "status #{pub['id']}")
  check(entry.at_css('[data-copy-bibtex]')&.[]('data-copy-bibtex') == pub['bibtex'], "citation #{pub['id']}")
  (pub['links'] || {}).each_value do |link|
    next unless link.start_with?('/')
    check(File.file?(File.join(site, URI.parse(link).path)), "local publication link #{link}")
  end
end
bib = File.read(File.join(site, 'publications.bib')).strip
check(bib == pubs.filter_map { |p| p['bibtex'] }.join("\n\n"), 'verbatim BibTeX concatenation')
keys = bib.scan(/^@\w+\{([^,]+),/).flatten
check(keys.size == pubs.count { |p| p['bibtex'] }, 'BibTeX entry count')
check(keys.uniq.size == keys.size, 'unique citation keys')
%w[llms.txt bio.md publications.json publications.bib].each do |name|
  text = File.read(File.join(site, name))
  check(!text.match?(/\{%|\{\{|<!DOCTYPE|<html/i), "Liquid/layout leakage in #{name}")
  check(!text.match?(/\.claude|CV - Bruno|AGENTS\.md|settings\.local/), "private metadata in #{name}")
  text.scan(/\]\((https:\/\/bruaristimunha\.github\.io[^)]*)\)/).flatten.each do |url|
    uri = URI(url)
    path = uri.path == '/' ? 'index.html' : uri.path.delete_prefix('/')
    check(File.file?(File.join(site, path)), "directory link #{url}")
    check(html.at_css("[id='#{uri.fragment}']"), "fragment #{url}") if uri.fragment
  end
end
%w[bio research-interests].each do |name|
  text = File.read(File.join(root, "_includes/#{name}.md")).strip
  check(File.read(File.join(site, 'bio.md')).include?(text), "shared Markdown #{name}")
end
private_paths = Dir.glob(File.join(site, '**', '*'), File::FNM_DOTMATCH).grep(/\.claude|CV - Bruno|AGENTS\.md|settings\.local/)
check(private_paths.empty?, 'private paths in built site')
puts "PASS: #{pubs.size} publication records and visible entries; #{keys.size} verbatim citations; statuses, titles, venues, local links, directory anchors, shared biography, no Liquid/layout/private leakage."

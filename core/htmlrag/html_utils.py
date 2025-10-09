import re
import bs4
from bs4 import Comment, BeautifulSoup, NavigableString, Tag


def replace_mathml_with_tex(soup: BeautifulSoup) -> None:
    for math_tag in soup.find_all('math'):
        tex_annotation = math_tag.find('annotation', encoding='application/x-tex')
        
        if tex_annotation:
            tex_code = tex_annotation.get_text(strip=True)
            is_display = math_tag.get('display') == 'block'
            if is_display:
                replacement_text = f"$${tex_code}$$"
            else:
                replacement_text = f"${tex_code}$"
            
            new_span = soup.new_tag('span', attrs={'class': 'math-formula'})
            new_span.string = replacement_text
            math_tag.replace_with(new_span)


MATH_SENTINEL = "§MATH§"
EXCLUDE_SECTIONS = {'references', 'bibliography', 'acknowledgments', 'acknowledgements'}
EQNUM_RE = re.compile(r'^\(?\d+[a-z]?\)?$')


def protect_dollar_math(soup: BeautifulSoup):
    import base64
    placeholders = {}
    counter = 0
    pat_block = re.compile(r'(?<!\$)(\$\$.*?\$\$)(?!\$)', re.S)
    pat_inline = re.compile(r'(?<!\$)(\$(?!\$).*?\$(?!\$))', re.S)
    skip = {'script', 'style', 'pre', 'code', 'textarea'}
    for node in list(soup.find_all(string=True)):
        parent = node.parent.name if node.parent else ''
        if parent in skip: continue
        text = str(node)
        def repl(m):
            nonlocal counter
            original = m.group(0)
            encoded = base64.b64encode(original.encode('utf-8')).decode('ascii')
            key = f"{MATH_SENTINEL}{counter}§{encoded}§"
            placeholders[key] = original
            counter += 1
            return key
        new_text = pat_block.sub(repl, text)
        new_text = pat_inline.sub(repl, new_text)
        if new_text != text:
            node.replace_with(NavigableString(new_text))
    return soup, placeholders

def restore_math(text: str, placeholders: dict) -> str:
    if not text:
        return text
    for k, v in placeholders.items():
        text = text.replace(k, v)
    return text

def table_has_display_or_dollars(table) -> bool:
    for node in table.find_all(string=True):
        parent = node.parent.name if node.parent else ''
        if parent in ('script', 'style'):
            continue
        s = str(node)
        if '\\displaystyle' in s:
            return True
        if re.search(r'\$\$.*?\$\$', s, flags=re.S):
            return True
    return False

def extract_formula_lines_from_table(table) -> list:
    lines = []
    for tr in table.find_all('tr'):
        cells = tr.find_all(['td', 'th'])
        if not cells:
            continue
        texts = [c.get_text(" ", strip=True) for c in cells]
        texts = [t for t in texts if t]
        if not texts:
            continue

        eqnum = ""
        if texts and EQNUM_RE.fullmatch(texts[-1]):
            eqnum = texts.pop()
            if not eqnum.startswith("("):
                eqnum = f"({eqnum})"

        line = " ".join(texts + ([eqnum] if eqnum else []))
        line = re.sub(r'\s+', ' ', line).strip()
        if line:
            lines.append(line)
    return lines

def merge_formula_tables_into_text(soup: BeautifulSoup) -> BeautifulSoup:
    for table in list(soup.find_all('table')):
        try:
            if not table_has_display_or_dollars(table):
                continue

            lines = extract_formula_lines_from_table(table)
            if not lines:
                continue

            formula_text = " " + " ".join(lines) + " "

            p_prev = None
            for prev_tag in table.find_all_previous(['p', 'div', 'li']):
                if prev_tag.get_text(strip=True):
                    p_prev = prev_tag
                    break

            p_next = None
            for next_tag in table.find_all_next(['p', 'div', 'li']):
                if next_tag.get_text(strip=True):
                    p_next = next_tag
                    break

            def is_ancestor(anc: Tag, node: bs4.Tag) -> bool:
                return anc is not None and node is not None and (anc in node.parents)

            if p_prev:
                prev_text = p_prev.get_text().rstrip()
                next_text = ""
                if p_next and not is_ancestor(p_next, p_prev):
                    next_text = " " + p_next.get_text().lstrip()
                    p_next.decompose()
                combined_text = prev_text + formula_text + next_text
                p_prev.clear()
                p_prev.append(NavigableString(combined_text))
                table.decompose()
            else:
                table.replace_with(NavigableString(formula_text))

        except Exception:
            pass
    return soup


def simplify_html(soup: BeautifulSoup, keep_attr: bool = False,
                   preserve_attrs: dict | None = None) -> tuple[str, dict]:
    for figure in soup.find_all('figure'):
        if figure.find('img'):
            figure.decompose()
    for img in soup.find_all('img'):
        img.decompose()

    replace_mathml_with_tex(soup)

    for script in soup(["script", "style"]):
        script.decompose()

    for note in soup.find_all(class_=['ltx_note', 'ltx_role_footnotemark']):
        note.decompose()
    for sup_tag in soup.find_all('sup'):
        sup_tag.decompose()

    for span in soup.find_all('span'):
        span.unwrap()

    soup = merge_formula_tables_into_text(soup)

    for a_tag in soup.find_all('a'):
        href = a_tag.get('href', '')
        if not href or href.startswith('#'):
            a_tag.replace_with(NavigableString(a_tag.get_text(separator=' ', strip=True)))

    for div in soup.find_all('div'):
        links = div.find_all('a')
        if len(links) > 3 and any("View original" in a.get_text() for a in links):
            div.decompose()

    for li_tag in soup.find_all('li'):
        if li_tag.contents and isinstance(li_tag.contents[0], NavigableString) and re.match(r'^\s*\d+\.\s*$', str(li_tag.contents[0])):
            li_tag.contents[0].extract()

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    if not keep_attr:
        preserve_attrs = preserve_attrs or {
            'td': {'colspan', 'rowspan'},
            'th': {'colspan', 'rowspan'},
        }
        for tag in soup.find_all(True):
            if tag.name in preserve_attrs:
                tag.attrs = {k: v for k, v in tag.attrs.items() if k in preserve_attrs[tag.name]}
            else:
                tag.attrs = {}

    while True:
        removed = False
        for tag in list(soup.find_all(True)):
            if tag.name in ['img', 'br']:
                continue
            if not tag.text.strip():
                tag.decompose()
                removed = True
        if not removed:
            break

    def concat_text(text):
        return re.sub(r'[\n\t ]+', '', text or '')

    structural_tags = {'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td'}
    for tag in list(soup.find_all(True)):
        if tag.name in structural_tags:
            continue
        children = [child for child in tag.contents if not isinstance(child, str)]
        if len(children) == 1:
            tag_text = tag.get_text()
            child_text = "".join([child.get_text() for child in tag.contents if not isinstance(child, str)])
            if concat_text(child_text) == concat_text(tag_text):
                tag.replace_with_children()

    soup, placeholders = protect_dollar_math(soup)

    res = str(soup)
    lines = [line for line in res.split("\n") if line.strip()]
    res = "\n".join(lines)
    return res, placeholders


def clean_html(html: str) -> tuple[str, dict]:
    soup = bs4.BeautifulSoup(html, 'html.parser')
    html_simplified, math_ph = simplify_html(soup)
    html_simplified = clean_xml(html_simplified)
    return html_simplified, math_ph

def clean_xml(html):
    html = re.sub(r"<\?xml.*?>", "", html)
    html = re.sub(r"<!DOCTYPE.*?>", "", html)
    html = re.sub(r"<!doctype.*?>", "", html)
    return html
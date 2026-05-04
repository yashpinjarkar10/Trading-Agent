import DOMPurify from 'dompurify';

/**
 * SafeHTML — render server-generated / LLM-generated markup without XSS risk.
 *
 * Bug #4 fix: every place that previously used `dangerouslySetInnerHTML`
 * with un-sanitized backend output should use this component instead. It runs
 * the input through DOMPurify with a strict allow-list (no <script>, no event
 * handlers, no javascript: URLs, no <iframe>).
 *
 * Industry-standard pattern (used by GitHub, Slack, Notion, etc. for any
 * user/AI-generated HTML).
 */

// Strict allow-list. We intentionally keep it small — chat / analysis output
// only needs basic formatting tags and links.
const ALLOWED_TAGS = [
  'a', 'b', 'br', 'code', 'em', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
  'hr', 'i', 'li', 'ol', 'p', 'pre', 'span', 'strong', 'table',
  'tbody', 'td', 'th', 'thead', 'tr', 'ul', 'div',
];

const ALLOWED_ATTR = ['href', 'target', 'rel', 'class', 'title'];

const PURIFY_CONFIG = {
  ALLOWED_TAGS,
  ALLOWED_ATTR,
  // Strip unknown protocols (only http/https/mailto allowed in <a href>)
  ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto):|[^a-z]|[a-z+.\-]+(?:[^a-z+.\-:]|$))/i,
  FORBID_TAGS: ['script', 'style', 'iframe', 'object', 'embed', 'form'],
  FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover', 'onfocus', 'onblur'],
};

// Force every <a> to open in new tab with rel="noopener noreferrer"
DOMPurify.addHook('afterSanitizeAttributes', (node) => {
  if (node.tagName === 'A') {
    node.setAttribute('target', '_blank');
    node.setAttribute('rel', 'noopener noreferrer');
  }
});

export function sanitizeHTML(html) {
  if (html == null) return '';
  return DOMPurify.sanitize(String(html), PURIFY_CONFIG);
}

export default function SafeHTML({ html, as: Tag = 'div', className }) {
  return (
    <Tag
      className={className}
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{ __html: sanitizeHTML(html) }}
    />
  );
}

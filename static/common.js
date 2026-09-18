"use strict";

/* Shared by both pages. This used to live twice, once inside each page's
   <script> block. The escaping helper in particular is the only thing between
   an attacker-supplied filename and stored XSS, so it must not be able to
   drift between the two copies. */

const el = (id) => document.getElementById(id);

/* Every value that reaches innerHTML goes through this. Filenames and error
   strings are attacker-controlled: anyone who can upload can plant markup. */
const ESCAPES = {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;","`":"&#96;"};
const esc = (v) => String(v == null ? "" : v).replace(/[&<>"'`]/g, (c) => ESCAPES[c]);

/* The access token: accepted from the URL once, then moved out of it. The two
   pages use different storage keys so an app token and an audit token cannot
   overwrite one another. */
function tokenStore(key){
  let value = sessionStorage.getItem(key) || "";
  const fromUrl = new URLSearchParams(location.search).get("token");
  if(fromUrl){
    value = fromUrl;
    sessionStorage.setItem(key, value);
    // Keep it out of history, bookmarks and any Referer we might emit.
    history.replaceState(null, "", location.pathname);
  }
  return {
    get: () => value,
    set: (next) => { value = next; sessionStorage.setItem(key, value); },
  };
}

/* A fetch wrapper that carries the token in a header — never a query string,
   so a hostile page cannot reach the API without a CORS preflight. Which status
   means "your token is no good" differs per page, and so does what to do about
   it: the audit page has three modes to re-render. */
function makeApi({header, store, isUnauthorised, onUnauthorised}){
  return async function api(path, opts={}){
    const headers = {...(opts.headers||{})};
    const token = store.get();
    if(token) headers[header] = token;
    const r = await fetch(path, {...opts, headers, credentials:"omit"});
    if(isUnauthorised(r.status)){
      onUnauthorised();
      throw new Error("unauthorised");
    }
    return r;
  };
}

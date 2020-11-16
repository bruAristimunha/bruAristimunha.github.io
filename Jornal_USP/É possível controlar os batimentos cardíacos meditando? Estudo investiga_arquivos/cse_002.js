(function(opts_){/*

 Copyright The Closure Library Authors.
 SPDX-License-Identifier: Apache-2.0
*/
var f=this||self,g=/^[\w+/_-]+[=]{0,2}$/,k=null,l=function(a){return(a=a.querySelector&&a.querySelector("script[nonce]"))&&(a=a.nonce||a.getAttribute("nonce"))&&g.test(a)?a:""},m=function(a,b){function c(){}c.prototype=b.prototype;a.prototype=new c;a.prototype.constructor=a},n=function(a){return a};var p=function(a){if(Error.captureStackTrace)Error.captureStackTrace(this,p);else{var b=Error().stack;b&&(this.stack=b)}a&&(this.message=String(a))};m(p,Error);p.prototype.name="CustomError";var r=function(a,b){a=a.split("%s");for(var c="",e=a.length-1,d=0;d<e;d++)c+=a[d]+(d<b.length?b[d]:"%s");p.call(this,c+a[e])};m(r,p);r.prototype.name="AssertionError";var t=function(a,b,c){if(!a){var e="Assertion failed";if(b){e+=": "+b;var d=Array.prototype.slice.call(arguments,2)}throw new r(""+e,d||[]);}},u=function(a,b){throw new r("Failure"+(a?": "+a:""),Array.prototype.slice.call(arguments,1));};var v=function(a,b){a:{try{var c=a&&a.ownerDocument,e=c&&(c.defaultView||c.parentWindow);e=e||f;if(e.Element&&e.Location){var d=e;break a}}catch(q){}d=null}if(d&&"undefined"!=typeof d[b]&&(!a||!(a instanceof d[b])&&(a instanceof d.Location||a instanceof d.Element))){d=typeof a;if("object"==d&&null!=a||"function"==d)try{var h=a.constructor.displayName||a.constructor.name||Object.prototype.toString.call(a)}catch(q){h="<object could not be stringified>"}else h=void 0===a?"undefined":null===a?"null":
typeof a;u("Argument is not a %s (or a non-Element, non-Location mock); got: %s",b,h)}};var w;var z=function(a,b){this.g=a===x&&b||"";this.h=y};z.prototype.toString=function(){return"Const{"+this.g+"}"};var A=function(a){if(a instanceof z&&a.constructor===z&&a.h===y)return a.g;u("expected object of type Const, got '"+a+"'");return"type_error:Const"},y={},x={};var D=function(a,b){this.f=a===B&&b||"";this.i=C};D.prototype.toString=function(){return"TrustedResourceUrl{"+this.f+"}"};
var E=function(a){if(a instanceof D&&a.constructor===D&&a.i===C)return a.f;var b=typeof a;u("expected object of type TrustedResourceUrl, got '"+a+"' of type "+("object"!=b?b:a?Array.isArray(a)?"array":b:"null"));return"type_error:TrustedResourceUrl"},H=function(a,b){var c=A(a);if(!F.test(c))throw Error("Invalid TrustedResourceUrl format: "+c);a=c.replace(aa,function(e,d){if(!Object.prototype.hasOwnProperty.call(b,d))throw Error('Found marker, "'+d+'", in format string, "'+c+'", but no valid label mapping found in args: '+
JSON.stringify(b));e=b[d];return e instanceof z?A(e):encodeURIComponent(String(e))});return G(a)},aa=/%{(\w+)}/g,F=/^((https:)?\/\/[0-9a-z.:[\]-]+\/|\/[^/\\]|[^:/\\%]+\/|[^:/\\%]*[?#]|about:blank#)/i,ba=/^([^?#]*)(\?[^#]*)?(#[\s\S]*)?/,J=function(a){var b=ca;a=H(da,a);a=E(a).toString();a=ba.exec(a);var c=a[3]||"";return G(a[1]+I("?",a[2]||"",b)+I("#",c,void 0))},C={},G=function(a){if(void 0===w){var b=null,c=f.trustedTypes;if(c&&c.createPolicy)try{b=c.createPolicy("goog#html",{createHTML:n,createScript:n,
createScriptURL:n})}catch(e){f.console&&f.console.error(e.message)}w=b}a=(b=w)?b.createScriptURL(a):a;return new D(B,a)},I=function(a,b,c){if(null==c)return b;if("string"===typeof c)return c?a+encodeURIComponent(c):"";for(var e in c)if(Object.prototype.hasOwnProperty.call(c,e)){var d=c[e];d=Array.isArray(d)?d:[d];for(var h=0;h<d.length;h++){var q=d[h];null!=q&&(b||(b=a),b+=(b.length>a.length?"&":"")+encodeURIComponent(e)+"="+encodeURIComponent(String(q)))}}return b},B={};var K=function(){this.c=""};K.prototype.toString=function(){return"SafeStyle{"+this.c+"}"};K.prototype.a=function(a){this.c=a};(new K).a("");var L=function(){this.b=""};L.prototype.a=function(a){this.b=a};L.prototype.toString=function(){return"SafeStyleSheet{"+this.b+"}"};(new L).a("");var ea=new z(x,"https://www.google.com/cse/static/style/look/%{versionDir}%{versionSlash}%{theme}.css"),M=new z(x,"https://www.google.com/cse/static/element/%{versionDir}%{versionSlash}default%{experiment}+%{lang}.css"),da=new z(x,"https://www.google.com/cse/static/element/%{versionDir}%{versionSlash}cse_element__%{lang}.js"),N=new z(x,"/");window.__gcse=window.__gcse||{};window.__gcse.ct=Date.now();
window.__gcse.scb=function(){var a=window.__gcse;O()||delete opts_.rawCss;var b=fa(a.initializationCallback||a.callback);google.search.cse.element.init(opts_)&&("explicit"!==a.parsetags?"complete"===document.readyState||"interactive"===document.readyState?(google.search.cse.element.go(),b&&b()):google.setOnLoadCallback(function(){google.search.cse.element.go();b&&b()},!0):b&&b())};function fa(a){return"function"===typeof a?a:"string"===typeof a&&"function"===typeof window[a]?window[a]:null}
function O(){return!(window.__gcse&&window.__gcse.plainStyle)}function P(a){var b=document.createElement("link");b.type="text/css";v(b,"HTMLLinkElement");b.rel="stylesheet";t(a instanceof D,'URL must be TrustedResourceUrl because "rel" contains "stylesheet"');b.href=E(a).toString();return b};var Q,ca=opts_.usqp?{usqp:opts_.usqp}:{},R=opts_.language.toLowerCase();Q=opts_.cselibVersion?J({versionDir:opts_.cselibVersion,versionSlash:N,lang:R}):J({versionDir:"",versionSlash:"",lang:R});var S=window.__gcse.scb,T=document.createElement("script");v(T,"HTMLScriptElement");T.src=E(Q);var U;var V=T.ownerDocument&&T.ownerDocument.defaultView;V&&V!=f?U=l(V.document):(null===k&&(k=l(f.document)),U=k);U&&T.setAttribute("nonce",U);T.type="text/javascript";S&&(T.onload=S);document.getElementsByTagName("head")[0].appendChild(T);
if(O()){document.getElementsByTagName("head")[0].appendChild(P(opts_.cselibVersion?H(M,{versionDir:opts_.cselibVersion,versionSlash:N,experiment:"",lang:opts_.language}):H(M,{versionDir:"",versionSlash:"",experiment:"",lang:opts_.language})));var W,X=opts_.uiOptions.cssThemeVersion||2,Y=opts_.theme.toLowerCase(),Z=X?"v"+X:Y.match(/v2_/g)?"v2":"",ha=Y.replace("v2_","");W=H(ea,{versionDir:Z,versionSlash:Z?N:"",theme:ha});document.getElementsByTagName("head")[0].appendChild(P(W))};
})({
  "cx": "012426026493194319923:hksl3buhmki",
  "language": "pt_PT",
  "theme": "V2_DEFAULT",
  "uiOptions": {
    "resultsUrl": "",
    "enableAutoComplete": true,
    "autoCompleteMatchType": "prefix",
    "enableImageSearch": true,
    "imageSearchLayout": "popup",
    "resultSetSize": "filtered_cse",
    "enableOrderBy": true,
    "orderByOptions": [{
      "label": "Relevance",
      "key": ""
    }, {
      "label": "Date",
      "key": "date"
    }],
    "overlayResults": false,
    "numTopRefinements": -1,
    "cssThemeVersion": 4,
    "isSafeSearchActive": false
  },
  "protocol": "https",
  "rawCss": ".gsc-control-cse{font-family:arial, sans-serif}.gsc-control-cse .gsc-table-result{font-family:arial, sans-serif}.gsc-refinementsGradient{background:linear-gradient(to left,rgba(233,233,233,1),rgba(233,233,233,0))}.gsc-control-cse{border-color:#FFFFFF;background-color:#FFFFFF}input.gsc-input,.gsc-input-box,.gsc-input-box-hover,.gsc-input-box-focus{border-color:#1094AB}.gsc-search-button-v2,.gsc-search-button-v2:hover,.gsc-search-button-v2:focus{border-color:#1094AB;background-color:#1094AB;background-image:none;filter:none}.gsc-search-button-v2 svg{fill:#FFFFFF}.gsc-tabHeader.gsc-tabhActive,.gsc-refinementHeader.gsc-refinementhActive{color:#1094AB;border-color:#1094AB;background-color:#FFFFFF}.gsc-tabHeader.gsc-tabhInactive,.gsc-refinementHeader.gsc-refinementhInactive{color:#1094AB;border-color:#1094AB;background-color:#E9E9E9}.gsc-webResult.gsc-result,.gsc-results .gsc-imageResult{border-color:#FFFFFF;background-color:#FFFFFF}.gsc-webResult.gsc-result:hover{border-color:#FFFFFF;background-color:#FFFFFF}.gs-webResult.gs-result a.gs-title:link,.gs-webResult.gs-result a.gs-title:link b,.gs-imageResult a.gs-title:link,.gs-imageResult a.gs-title:link b{color:#0000CC}.gs-webResult.gs-result a.gs-title:visited,.gs-webResult.gs-result a.gs-title:visited b,.gs-imageResult a.gs-title:visited,.gs-imageResult a.gs-title:visited b{color:#0000CC}.gs-webResult.gs-result a.gs-title:hover,.gs-webResult.gs-result a.gs-title:hover b,.gs-imageResult a.gs-title:hover,.gs-imageResult a.gs-title:hover b{color:#0000CC}.gs-webResult.gs-result a.gs-title:active,.gs-webResult.gs-result a.gs-title:active b,.gs-imageResult a.gs-title:active,.gs-imageResult a.gs-title:active b{color:#0000CC}.gsc-cursor-page{color:#0000CC}a.gsc-trailing-more-results:link{color:#0000CC}.gs-webResult:not(.gs-no-results-result):not(.gs-error-result) .gs-snippet,.gs-fileFormatType{color:#000000}.gs-webResult div.gs-visibleUrl{color:#008000}.gs-webResult div.gs-visibleUrl-short{color:#008000}.gsc-cursor-box{border-color:#FFFFFF}.gsc-results .gsc-cursor-box .gsc-cursor-page{border-color:#1094AB;background-color:#E9E9E9;color:#1094AB}.gsc-results .gsc-cursor-box .gsc-cursor-current-page{border-color:#1094AB;background-color:#FFFFFF;color:#1094AB}.gsc-webResult.gsc-result.gsc-promotion{border-color:#336699;background-color:#FFFFFF}.gsc-completion-title{color:#0000CC}.gsc-completion-snippet{color:#000000}.gs-promotion a.gs-title:link,.gs-promotion a.gs-title:link *,.gs-promotion .gs-snippet a:link{color:#0000CC}.gs-promotion a.gs-title:visited,.gs-promotion a.gs-title:visited *,.gs-promotion .gs-snippet a:visited{color:#0000CC}.gs-promotion a.gs-title:hover,.gs-promotion a.gs-title:hover *,.gs-promotion .gs-snippet a:hover{color:#0000CC}.gs-promotion a.gs-title:active,.gs-promotion a.gs-title:active *,.gs-promotion .gs-snippet a:active{color:#0000CC}.gs-promotion .gs-snippet,.gs-promotion .gs-title .gs-promotion-title-right,.gs-promotion .gs-title .gs-promotion-title-right *{color:#000000}.gs-promotion .gs-visibleUrl,.gs-promotion .gs-visibleUrl-short{color:#008000}.gcsc-find-more-on-google{color:#0000CC}.gcsc-find-more-on-google-magnifier{fill:#0000CC}",
  "cse_token": "AJvRUv0_zwFHM0OIhsBVKgMpo-Wf:1605555042339",
  "isHostedPage": false,
  "exp": ["csqr", "cc"],
  "cselibVersion": "83175b7bfe796f29",
  "usqp": "CAI\u003d"
});
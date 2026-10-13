Vendored files from @elevenlabs/client 1.27.0
=============================================

Package:  @elevenlabs/client
Version:  1.27.0
License:  MIT (copied unchanged as LICENSE in this directory; Copyright (c) 2025 ElevenLabs)
Source:   https://registry.npmjs.org/@elevenlabs/client/-/client-1.27.0.tgz
          (npm pack @elevenlabs/client@1.27.0)
Tarball integrity (npm registry dist.integrity, verified):
          sha512-Sy2gATXyPwkQZhJC+CGibHK0LaO6PdPyy6oEn1FgcaEV4W/Q/KkdUGVt4EX99WWpIArZF2PIRjpqWtRgwKCWtA==

The files below are copied byte for byte from the tarball's package/ directory. They are also
byte-identical to https://cdn.jsdelivr.net/npm/@elevenlabs/client@1.27.0/worklets/<file>.

sha256                                                            file (tarball path)
6731b766e9e6176d56b9c6964975dd7dd045db88ce3b2b0b787f5fbeb9f1325d  rawAudioProcessor.js     (package/worklets/rawAudioProcessor.js)
e352f9c0e6e540baed2c16247015b997aa9f46767b3c6b0fc41afd3a5053870b  audioConcatProcessor.js  (package/worklets/audioConcatProcessor.js)
0bfed2a2aa8d106bdb144f55416e193450fbc1e1b3257e3852bc64418069de5a  LICENSE                  (package/LICENSE)

Verify:  cd src/studiodesk/static/vendor/elevenlabs-client-1.27.0 && shasum -a 256 *.js LICENSE

These are AudioWorklet modules passed to Conversation.startSession({ workletPaths }) in
static/voice.js. The main SDK bundle (dist/lib.iife.js, about 1.1 MB) is not vendored; it is
still loaded from jsDelivr with Subresource Integrity (see static/voice.html).

To upgrade: npm pack the new version, copy the same files into a new
vendor/elevenlabs-client-<version>/ directory, update this README's hashes, and change
SDK_VERSION in voice.js together with the <script> tag (URL and integrity) in voice.html.
Do not edit these files in place.

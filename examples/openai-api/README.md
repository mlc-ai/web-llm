### OpenAI API Demos

Run `npm install` first, followed by `npm start`.

To run different scripts, you can modify `package.json` from the default 
```json
"scripts": {
    "start": "parcel src/openai_api.html  --port 8888",
    "build": "parcel build src/openai_api.html --dist-dir lib"
},
```

to, say
```json
"scripts": {
    "start": "parcel src/seed.html  --port 8888",
    "build": "parcel build src/seed.html --dist-dir lib"
},
```
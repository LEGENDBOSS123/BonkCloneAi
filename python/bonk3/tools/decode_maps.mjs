// Decode bonk.io map database strings -> MapData JSON, using the real game's
// codec (bonk-map, via the bundled demo/map-loader.js). Run offline; the Rust
// engine and the Python renderer both consume the output directly.
//
//   node python/bonk3/tools/decode_maps.mjs --list
//   node python/bonk3/tools/decode_maps.mjs --name "gang grounds 2.0"
//   node python/bonk3/tools/decode_maps.mjs --index 322 --out maps/gang-grounds.json
//   node python/bonk3/tools/decode_maps.mjs --all --max 50
//
// YOUR OWN MAP — paste the share code the game gives you, or put it in a file:
//   node python/bonk3/tools/decode_maps.mjs --string "ILAcCFgRWBhKDGBzEq3..."
//   node python/bonk3/tools/decode_maps.mjs --code-file mymap.txt --out maps/mine.json
//
// Maps land in python/bonk3/maps/ as <slug>.json.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '../../..');
const ENV_DIR = path.join(REPO, 'bonk-recreation/bonk-enviroment');
const MAPS_DB = path.join(ENV_DIR, 'maps.json');
const OUT_DIR = path.join(REPO, 'python/bonk3/maps');

// map-loader.js is an IIFE that keeps decodeFromDatabase private; re-expose it.
globalThis.window = globalThis;
const loader = fs.readFileSync(path.join(ENV_DIR, 'demo/map-loader.js'), 'utf8');
(0, eval)(loader.replace(/\}\)\(\);\s*$/, '  globalThis.__decode = decodeFromDatabase;\n})();'));
const decode = globalThis.__decode;
if (!decode) throw new Error('map-loader.js did not expose decodeFromDatabase');

const db = JSON.parse(fs.readFileSync(MAPS_DB, 'utf8'));

const args = process.argv.slice(2);
const flag = (n) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : null; };
const has = (n) => args.includes(n);

const slug = (s) => (s || 'map').toLowerCase().trim()
    .replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60) || 'map';

function tryDecode(i) {
    try {
        const m = decode(db[i]);
        if (!m?.physics?.shapes?.length) return null;   // empty/corrupt entry
        return m;
    } catch { return null; }
}

function summarize(i, m) {
    return {
        index: i,
        name: m.metadata.name,
        author: m.metadata.author,
        dbid: m.metadata.databaseId,
        shapes: m.physics.shapes.length,
        bodies: m.physics.bodies.length,
        joints: m.physics.joints.length,
        spawns: m.spawns.length,
        ppm: m.physics.pixelsPerMeter,
    };
}

function write(i, m) {
    fs.mkdirSync(OUT_DIR, { recursive: true });
    const name = flag('--out') || path.join(OUT_DIR, `${slug(m.metadata.name)}.json`);
    fs.writeFileSync(name, JSON.stringify(m));
    console.log(`wrote ${name}  (${m.physics.shapes.length} shapes, ` +
                `${m.physics.bodies.length} bodies, ${m.spawns.length} spawns)`);
}

// A pasted bonk share code / database string — the "my own map" path.
const codeArg = flag('--string')
    ?? (flag('--code-file') ? fs.readFileSync(flag('--code-file'), 'utf8').trim() : null);

if (codeArg) {
    let m;
    try {
        m = decode(codeArg);
    } catch (e) {
        console.error(`could not decode that string: ${e.message}`);
        console.error('Expect the map code bonk gives you (a long base64-ish blob), ' +
                      'not a URL and not a JSON file.');
        process.exit(1);
    }
    if (!m?.physics?.shapes?.length) {
        console.error('decoded, but the map has no shapes — wrong string?');
        process.exit(1);
    }
    console.log(JSON.stringify(summarize(-1, m)));
    write(-1, m);
} else if (has('--list')) {
    const rows = [];
    for (let i = 0; i < db.length; i++) {
        const m = tryDecode(i);
        if (m) rows.push(summarize(i, m));
    }
    rows.sort((a, b) => a.shapes - b.shapes);
    console.table(rows.slice(0, Number(flag('--max') || 40)));
    console.log(`${rows.length} decodable maps of ${db.length}`);
} else if (flag('--name')) {
    const want = flag('--name').toLowerCase();
    let hits = 0;
    for (let i = 0; i < db.length && hits < Number(flag('--max') || 1); i++) {
        const m = tryDecode(i);
        if (m && m.metadata.name.toLowerCase().includes(want)) {
            console.log(JSON.stringify(summarize(i, m)));
            write(i, m);
            hits++;
        }
    }
    if (!hits) console.log(`no map matching ${JSON.stringify(want)}`);
} else if (flag('--index')) {
    const i = Number(flag('--index'));
    const m = tryDecode(i);
    if (!m) throw new Error(`map ${i} did not decode`);
    console.log(JSON.stringify(summarize(i, m)));
    write(i, m);
} else if (has('--all')) {
    const max = Number(flag('--max') || db.length);
    let n = 0;
    for (let i = 0; i < db.length && n < max; i++) {
        const m = tryDecode(i);
        if (!m) continue;
        fs.mkdirSync(OUT_DIR, { recursive: true });
        fs.writeFileSync(path.join(OUT_DIR, `${String(i).padStart(4, '0')}-${slug(m.metadata.name)}.json`),
                         JSON.stringify(m));
        n++;
    }
    console.log(`wrote ${n} maps to ${OUT_DIR}`);
} else {
    console.log('usage:\n' +
      '  --string "<bonk map code>"   decode YOUR map from a share code\n' +
      '  --code-file FILE             same, code read from a file\n' +
      '  --name "gang grounds"        find one in the bundled 2137-map database\n' +
      '  --index N                    take database entry N\n' +
      '  --list [--max N]             browse the database (sorted by complexity)\n' +
      '  --all [--max N]              dump many at once\n' +
      '  --out FILE                   where to write (default python/bonk3/maps/<slug>.json)');
}

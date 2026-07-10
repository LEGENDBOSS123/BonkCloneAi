import { createApp } from "./app.mjs";
import { Input } from "./input.mjs";
import { MapRenderingContext } from "./map/index.mjs";
import { loadBonkMap } from "./bonkmap/bonkmap_parser.mjs";

const WORLD_HEIGHT = 64; // meters visible vertically (width follows the screen)

const app = await createApp();
const input = new Input();

// Load and parse a Bonk.io map into our Map.
const map = await loadBonkMap(new URL("./bonkmap/map1.json", import.meta.url), { gravity: 20 });
top.map = map;
// The rendering context steps the map at 30 TPS, follows the player with a
// camera that shows WORLD_HEIGHT meters of vertical world space, and
// interpolates for display.
const ctx = new MapRenderingContext(map, {
    stage: app.stage,
    screen: app.screen,
    worldHeight: WORLD_HEIGHT,
    interpolate: true,
});
app.ticker.add((ticker) => ctx.update(ticker.deltaMS / 1000, input));

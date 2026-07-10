// Tracks which keys are currently held. Add more codes to TRACKED as the game
// grows. Query with isDown(code) each frame.
const TRACKED = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "KeyX"]);

export class Input {
    #held = new Set();

    constructor(target = window) {
        target.addEventListener("keydown", (e) => {
            if (!TRACKED.has(e.code)) return;
            this.#held.add(e.code);
            e.preventDefault(); // stop arrow keys from scrolling the page
        });
        target.addEventListener("keyup", (e) => this.#held.delete(e.code));
    }

    isDown(code) {
        return this.#held.has(code);
    }
}

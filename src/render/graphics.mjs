// Adaptive Graphics: real Pixi in the browser, a chainable no-op stub headless.
//
// The simulation entities (Player, Shape) build their views in their
// constructors, which drags pixi.js into any import of the sim. Headless (Node)
// training doesn't render, so there we substitute a stub that accepts the whole
// Graphics API surface (circle().fill(), stroke(), rect(), addChild, clear,
// position.set, alpha, ...) as chainable no-ops while keeping the few fields
// that are READ by sim code working (Shape.x/y come from view.position).

class StubGraphicsBase {
    constructor() {
        this.x = 0;
        this.y = 0;
        this.rotation = 0;
        this.alpha = 1;
        this.children = [];
        const self = this;
        this.position = { set(x = 0, y = 0) { self.x = x; self.y = y; } };
        this.scale = { set() { } };
    }

    addChild(child) {
        this.children.push(child);
        return child;
    }
}

const stubHandler = {
    get(target, prop, receiver) {
        if (prop in target) return Reflect.get(target, prop, receiver);
        // Any unknown property is assumed to be a drawing method: swallow the
        // call, return the proxy so chains like circle().fill() keep working.
        return () => receiver;
    },
};

function StubGraphics() {
    return new Proxy(new StubGraphicsBase(), stubHandler);
}

let Graphics;
if (typeof document !== "undefined") {
    ({ Graphics } = await import("pixi.js"));
} else {
    Graphics = StubGraphics;
}

export { Graphics };

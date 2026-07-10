import { Graphics } from "../render/graphics.mjs";

// Base class for platform shapes. Holds the shared visual + physical settings
// and the Pixi view. Subclasses (Rectangle, Polygon, ...) implement draw() for
// their geometry and vertices() for collision.
export class Shape {
    constructor({ x = 0, y = 0, color = 0x3a5a40, bounciness = 0, friction = 0, solid = true } = {}) {
        this.color = color;
        this.bounciness = bounciness; // 0 = no bounce, 1 = perfectly elastic
        this.friction = friction; // 0 = frictionless ice, 1 = grippy
        this.solid = solid; // false = rendered only, no collider (decorative)
        this.view = new Graphics();
        this.view.position.set(x, y);
    }

    // Clear and re-render. Subclasses call this once their geometry fields are
    // set; call again whenever geometry or color changes.
    redraw() {
        this.view.clear();
        this.draw();
        return this;
    }

    // Subclasses fill this.view with their geometry (local coords), using
    // this.color.
    draw() {
        throw new Error(`${this.constructor.name} must implement draw()`);
    }

    // World-space vertices of the outline, for collision. Subclasses override.
    vertices() {
        throw new Error(`${this.constructor.name} must implement vertices()`);
    }

    // Engine-neutral collision descriptor(s) for this shape's geometry, in
    // coordinates local to the body placed at (this.x, this.y). The physics layer
    // turns these into its own colliders and applies this.friction /
    // this.bounciness. Returns an array of:
    //   { type: "circle", radius, cx?, cy? }
    //   { type: "box", hx, hy, cx?, cy?, angle? }
    //   { type: "polygon", vertices: [{ x, y }, ...] }   // convex, local coords
    // Subclasses override.
    collisionShapes() {
        throw new Error(`${this.constructor.name} must implement collisionShapes()`);
    }

    get x() {
        return this.view.x;
    }

    get y() {
        return this.view.y;
    }
}

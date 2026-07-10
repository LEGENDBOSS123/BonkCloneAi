import { Shape } from "./shape.mjs";

// An axis-aligned rectangular platform. The shape's (x, y) origin is its
// top-left corner.
export class Rectangle extends Shape {
    constructor({ width = 120, height = 24, ...opts } = {}) {
        super(opts);
        this.width = width;
        this.height = height;
        this.redraw();
    }

    draw() {
        this.view.rect(0, 0, this.width, this.height).fill(this.color);
    }

    vertices() {
        const { x, y } = this.view;
        return [
            { x, y },
            { x: x + this.width, y },
            { x: x + this.width, y: y + this.height },
            { x, y: y + this.height },
        ];
    }

    // Box colliders are centered, so use half-extents and shift the collider to
    // the rectangle's center (origin is the top-left corner).
    collisionShapes() {
        return [{ type: "box", hx: this.width / 2, hy: this.height / 2, cx: this.width / 2, cy: this.height / 2 }];
    }
}

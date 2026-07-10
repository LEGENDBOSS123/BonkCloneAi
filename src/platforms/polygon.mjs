import { Shape } from "./shape.mjs";

// An arbitrary polygon platform. `points` are local coordinates relative to the
// shape's (x, y) origin: [{ x, y }, ...].
export class Polygon extends Shape {
    constructor({ points = [], ...opts } = {}) {
        super(opts);
        if (points.length < 3) {
            throw new Error("Polygon needs at least 3 points");
        }
        this.points = points;
        this.redraw();
    }

    draw() {
        this.view.poly(this.points.flatMap((p) => [p.x, p.y])).fill(this.color);
    }

    vertices() {
        const { x, y } = this.view;
        return this.points.map((p) => ({ x: x + p.x, y: y + p.y }));
    }

    // Points are already local to the origin. Box2D computes the convex hull
    // itself (and caps it at 8 vertices), so hand them over directly.
    collisionShapes() {
        return [{ type: "polygon", vertices: this.points }];
    }
}

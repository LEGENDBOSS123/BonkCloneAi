import { Shape } from "./shape.mjs";

// A circular platform. The shape's (x, y) origin is the circle's center.
export class Circle extends Shape {
  constructor({ radius = 40, segments = 24, ...opts } = {}) {
    super(opts);
    this.radius = radius;
    this.segments = segments; // vertex count returned by vertices() for collision
    this.redraw();
  }

  draw() {
    this.view.circle(0, 0, this.radius).fill(this.color);
  }

  // Approximate the circle as an N-gon in world space for polygon-based
  // collision. Use this.center for true circle-vs-X tests.
  vertices() {
    const { x, y } = this.view;
    const out = [];
    for (let i = 0; i < this.segments; i++) {
      const a = (i / this.segments) * Math.PI * 2;
      out.push({ x: x + Math.cos(a) * this.radius, y: y + Math.sin(a) * this.radius });
    }
    return out;
  }

  get center() {
    return { x: this.view.x, y: this.view.y };
  }

  // Origin is the center, so a plain circle collider lines up with the body.
  collisionShapes() {
    return [{ type: "circle", radius: this.radius }];
  }
}

import { Graphics } from "../render/graphics.mjs";

// The player's circle. A dynamic body driven by forces (arrow keys) and an
// impulse (jump). Gravity is applied by the physics world. Jumping is only
// allowed when a short downward raycast finds ground beneath the circle.
export class Player {
    constructor({ x = 0, y = 0, radius = 1, color = 0x00e5ff, controllable = true, moveHorizontalAccel = 12, moveVerticalAccel = 12, jumpSpeed = 10, jumpVelocityThreshold = 4, groundMargin = 0.15 } = {}) {
        this.x = x;
        this.y = y;
        this.spawnx = x;
        this.spawny = y;
        this.heavyPower = 1000;
        this.radius = radius;
        this.controllable = controllable; // if false, ignores input and just sits (still affected by physics)
        this.moveHorizontalAccel = moveHorizontalAccel; // horizontal acceleration, units/s^2 (mass-scaled to force)
        this.moveVerticalAccel = moveVerticalAccel; // vertical acceleration, units/s^2 (mass-scaled to force)
        this.jumpSpeed = jumpSpeed; // upward launch velocity, units/s (mass-scaled to impulse)
        this.jumpVelocityThreshold = jumpVelocityThreshold; // minimum downward velocity to allow jumping
        this.groundMargin = groundMargin; // how far past the bottom of the circle the ground ray reaches (world units)
        // Wired up by attach().
        this.body = null;
        this.collider = null;
        this.physics = null;
        this.currentInput = null;

        this.view = new Graphics().circle(0, 0, radius).fill(color);
        // Border whose opacity tracks how "heavy" the player is. Drawn as a child
        // of the view so it follows the player; alpha updated each tick from mass.
        this.border = new Graphics().circle(0, 0, radius).stroke({ color: 0xffffff, width: radius * 0.18 });
        this.border.alpha = 0;
        this.view.addChild(this.border);
        this.view.position.set(x, y);
    }

    // Mass ranges from 1 (base) to 4.7 (fully heavy). Map that to border opacity
    // 0..1 so the outline fades in as the player gets heavier.
    updateBorder() {
        if (!this.body) return;
        const mass = this.body.mass() || 1;
        this.border.alpha = Math.min(Math.max((mass - 1) / 3.7, 0), 1);
    }

    // Wire up the physics pieces the Map created for this player.
    attach({ body, collider, physics }) {
        this.body = body;
        this.collider = collider;
        this.physics = physics;
        return this;
    }

    getInput(rawinput) {
        return {
            up: rawinput.isDown("ArrowUp"),
            down: rawinput.isDown("ArrowDown"),
            left: rawinput.isDown("ArrowLeft"),
            right: rawinput.isDown("ArrowRight"),
            heavy: rawinput.isDown("KeyX"),
        }
    }

    // Drive the body each tick. Runs before the world steps.
    control(rawinput) {
        if (!this.body || !this.controllable) return;
        let input = this.getInput(rawinput);
        this.currentInput = input;
        this.body.resetForces(true);
        let dirX = 0;
        let dirY = 0;

        if (input.left) dirX -= 1;
        if (input.right) dirX += 1;
        if (input.down) dirY += 1;
        if (input.up) dirY -= 1;
        if (input.heavy) {
            this.heavyPower = Math.max(0, this.heavyPower - 10);
        }
        else {
            this.heavyPower = Math.min(1000, this.heavyPower + 5);
        }

        // RigidBody has no setMass; use setAdditionalMass, which stacks on top of
        // the collider's 1 kg. So additional = desiredTotal - 1.
        if (input.heavy) {
            this.body.setAdditionalMass(3.7 * this.heavyPower/1000, true);
        }
        else {
            this.body.setAdditionalMass(0, true);
        }

        if (dirX !== 0 || dirY !== 0) this.body.addForce({ x: dirX * this.moveHorizontalAccel, y: dirY * this.moveVerticalAccel }, true);

        if (input.up && this.isGrounded()) this.jump();

        this.updateBorder();
    }

    isGrounded() {
        const pos = this.body.translation();
        const hit = this.physics.castRay(
            { x: pos.x, y: pos.y },
            { x: 0, y: 1 },
            this.radius + this.groundMargin,
            { excludeCollider: this.collider }
        );
        return hit !== null;
    }

    jump() {
        let velocity = this.body.linvel();
        if (Math.abs(velocity.y) >= this.jumpVelocityThreshold) return;
        this.body.applyImpulse({ x: 0, y: -this.jumpSpeed * this.body.mass() }, true);
    }
}

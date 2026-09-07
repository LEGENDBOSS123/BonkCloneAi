if (top.a) clearInterval(top.a);

let isRestarting = false;

top.playerQueue = top.playerQueue || [];
top.knownPlayers = top.knownPlayers || {};

top.a = setInterval(() => {
    try {
        const doc = top.Gdocument;
        const gameRenderer = doc ? doc.getElementById("gamerenderer") : null;

        if (!gameRenderer) return;
        if (!top.playerids || typeof top.RECIEVE !== "function") return;

        const playerKeys = Object.keys(top.playerids);

        // ------------------------------------------------------------
        // Detect new players -> add to END of queue
        // ------------------------------------------------------------

        const currentPlayers = {};

        for (const id of playerKeys) {
            if (String(id) === String(top.myid)) continue;

            const player = top.playerids[id];
            if (!player) continue;

            currentPlayers[String(id)] = true;

            if (!top.knownPlayers[String(id)]) {
                top.knownPlayers[String(id)] = true;

                if (!top.playerQueue.includes(String(id))) {
                    top.playerQueue.push(String(id));
                }
            }
        }

        // ------------------------------------------------------------
        // Remove players who left
        // ------------------------------------------------------------

        top.playerQueue = top.playerQueue.filter(id =>
            currentPlayers[String(id)]
        );

        for (const id of Object.keys(top.knownPlayers)) {
            if (!currentPlayers[String(id)]) {
                delete top.knownPlayers[id];
            }
        }

        // ------------------------------------------------------------
        // Count Team 1 players
        // ------------------------------------------------------------

        let team1Count = 0;

        for (const id of playerKeys) {
            const player = top.playerids[id];

            if (player && player.team === 1) {
                team1Count++;
            }
        }

        // ------------------------------------------------------------
        // Determine if matchmaking is needed
        //
        // children = 0:
        //     Game is over -> matchmaking
        //
        // children = 1:
        //     Game is running, but if Team 1 has < 2 players,
        //     matchmaking is needed.
        //
        // children >= 2:
        //     Game is running normally -> do nothing
        // ------------------------------------------------------------

        const childCount = gameRenderer.children.length;

        let shouldTrigger = false;

        if (childCount === 0) {
            shouldTrigger = true;
        } else if (childCount === 1 && team1Count < 2) {
            shouldTrigger = true;
        }

        if (!shouldTrigger) {
            isRestarting = false;
            return;
        }

        // ------------------------------------------------------------
        // No players available -> do nothing
        // ------------------------------------------------------------

        if (top.playerQueue.length === 0) {
            return;
        }

        if (isRestarting) return;

        // ============================================================
        // MATCHMAKING
        // ============================================================

        // Take FIRST player in queue
        const chosenId = top.playerQueue.shift();

        // Put them at END of queue
        top.playerQueue.push(String(chosenId));

        isRestarting = true;

        // ------------------------------------------------------------
        // Everyone except YOU and chosen player -> Team 0
        // ------------------------------------------------------------

        for (const id of playerKeys) {
            if (String(id) === String(top.myid)) continue;
            if (String(id) === String(chosenId)) continue;

            const idVal = isNaN(id) ? id : Number(id);

            top.RECIEVE(
                "42" + JSON.stringify([
                    18,
                    idVal,
                    0
                ])
            );
        }

        // ------------------------------------------------------------
        // YOU -> Team 1
        // ------------------------------------------------------------

        const myIdVal = isNaN(top.myid)
            ? top.myid
            : Number(top.myid);

        top.RECIEVE(
            "42" + JSON.stringify([
                18,
                myIdVal,
                1
            ])
        );

        // ------------------------------------------------------------
        // CHOSEN PLAYER -> Team 1
        // ------------------------------------------------------------

        const chosenVal = isNaN(chosenId)
            ? chosenId
            : Number(chosenId);

        top.RECIEVE(
            "42" + JSON.stringify([
                18,
                chosenVal,
                1
            ])
        );

        // ------------------------------------------------------------
        // Start/restart game
        // ------------------------------------------------------------

        if (typeof top.startGame === "function") {
            setTimeout(() => {
                try {
                    top.startGame();
                } catch (err) {}

                isRestarting = false;
            }, 500);
        } else {
            isRestarting = false;
        }

    } catch (err) {}
}, 2000);
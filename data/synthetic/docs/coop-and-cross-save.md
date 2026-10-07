# Co-op and Cross-Save Guide

How to play Starfall Outpost together across platforms and carry your progress between them.

## Co-op basics

Up to 8 players can share an outpost since the Outpost Alliance update (1.3.0); earlier versions allowed 4. One player hosts the session and the world is saved on the host's device. Guests keep their character, inventory and personal research, while buildings, reactors and shared research belong to the host's world.

To host, open **Play → Host Co-op** and choose a privacy setting: **Public**, **Friends Only** or **Invite Only**. To join, accept an invite, use **Play → Join Friend** or browse public sessions with **Quick Play**.

## Cross-play

Cross-play works between all four platforms: PC, PS5, Xbox Series X|S and Switch. To add a friend on another platform, link both accounts to a Starfall ID and add each other under **Social → Starfall Friends**.

Known cross-play issues and fixes:

- Switch friend invites failing with error **SO-4012** were fixed in 1.0.1.
- Joining a PS5 host with **Friends Only** privacy failed with **SO-2207 Not permitted** for Starfall friends on other platforms; 1.2.1 made Friends Only lobbies accept cross-platform Starfall friends.
- Switch players could not find public Outpost Alliance sessions in some regions after 1.3.0; set privacy to Public and use Join Friend as a workaround while this is investigated.

## Host migration

If the host leaves, another player becomes host automatically and the session continues from the latest world state. In 1.0.0 the oxygen reactor output did not replicate after host migration, so everyone could suffocate even with a full reactor; this was fixed in 1.0.1. An item duplication exploit through cargo crate transfers during host migration was fixed in 1.2.0.

## Cross-save

Cross-save, added in Deep Core (1.2.0), lets you continue the same worlds on any platform. Link each platform account to the same Starfall ID under **Settings → Account → Cross-Save**. Your saves upload to Starfall cloud storage when you quit to the main menu and download when you start the game.

When both the device and the cloud have changed since the last sync, a conflict dialog shows the time and playtime of each save:

- **Keep Local** keeps the save on this device and uploads it to the cloud.
- **Keep Cloud** downloads the cloud save and replaces the save on this device.

In 1.2.0 these two buttons were swapped, so Keep Local could load and upload an older cloud save. Version 1.2.1 fixed the dialog. If you lost progress this way on 1.2.0, open **Load Game → Restore Backup** on the device that had the newer save before syncing again.

## Tips for smooth sessions

- The host should have the best connection and hardware; on PS5, hosting 8 players works best in Performance mode on 1.3.1 or later.
- Trade terminals are locked to one player at a time since 1.3.1; wait for the other player to close it.
- Lightning rods protect anchored structures within 30 m since 1.4.1, so place them before a storm season in shared bases.

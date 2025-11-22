const { expect } = require("chai");

describe("Tracker", function () {
  it("allows owner to set authorized and record updates", async function () {
    const [owner, user] = await ethers.getSigners();
    const Tracker = await ethers.getContractFactory("Tracker");
    const tracker = await Tracker.deploy();
    await tracker.deployed();

    expect(await tracker.authorized(user.address)).to.equal(false);
    await tracker.setAuthorized(user.address, true);
    expect(await tracker.authorized(user.address)).to.equal(true);

    const tx = await tracker.connect(user).updateItem("ITEM123", 1, "Packed at vendor");
    const rcpt = await tx.wait();

    const ev = rcpt.events.find(e => e.event === "ItemUpdated");
    expect(ev).to.not.be.undefined;
    expect(ev.args.itemId).to.equal("ITEM123");

    const latest = await tracker.getLatest("ITEM123");
    expect(latest.status).to.equal(1);
    expect(latest.note).to.equal("Packed at vendor");
  });
});

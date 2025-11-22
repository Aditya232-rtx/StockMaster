// Simple test to register item 789
const hre = require("hardhat");

async function main() {
    const Tracker = await hre.ethers.getContractFactory("Tracker");
    const tracker = Tracker.attach("0x5FC8d32690cc91D4c39d9d3abcBD16989F875707");

    const itemId = "789";
    const hashInput = JSON.stringify({ itemId });
    const hash = hre.ethers.keccak256(hre.ethers.toUtf8Bytes(hashInput));

    console.log("Registering item:", itemId);
    console.log("Hash:", hash);

    const tx = await tracker.registerItem(itemId, hash);
    console.log("Transaction:", tx.hash);
    await tx.wait();
    console.log("✅ Registered!");

    // Verify
    const storedHash = await tracker.getItemHash(itemId);
    console.log("Stored hash:", storedHash);
    console.log("Match:", storedHash === hash);
}

main().catch(err => { console.error(err); process.exit(1); });

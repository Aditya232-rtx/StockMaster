// Script to check registered items on the blockchain
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
    // Load deployment data
    const deploymentsPath = path.join(__dirname, "..", "deployments.json");
    const deployments = JSON.parse(fs.readFileSync(deploymentsPath, "utf8"));

    // Get latest localhost deployment
    const latestDeployment = deployments
        .filter(d => d.network === 'localhost')
        .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0];

    if (!latestDeployment) {
        console.log("No localhost deployment found");
        return;
    }

    console.log("Contract Address:", latestDeployment.contractAddress);

    // Get contract
    const Tracker = await hre.ethers.getContractFactory("Tracker");
    const tracker = Tracker.attach(latestDeployment.contractAddress);

    // Check if item is registered
    const itemId = process.argv[2];
    if (!itemId) {
        console.log("Usage: node checkItem.js <itemId>");
        return;
    }

    console.log("\nChecking item:", itemId);

    try {
        const hash = await tracker.getItemHash(itemId);
        console.log("Item Hash:", hash);

        if (hash === "0x0000000000000000000000000000000000000000000000000000000000000000") {
            console.log("❌ Item NOT registered");
        } else {
            console.log("✅ Item IS registered");
            const isRegistered = await tracker.isItemRegistered(hash);
            console.log("Hash registered:", isRegistered);
        }
    } catch (err) {
        console.log("Error:", err.message);
    }
}

main().catch(err => { console.error(err); process.exit(1); });

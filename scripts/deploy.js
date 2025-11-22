const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying contracts with account:", deployer.address);

  const Tracker = await hre.ethers.getContractFactory("Tracker");
  const tracker = await Tracker.deploy();
  await tracker.waitForDeployment();

  const contractAddress = await tracker.getAddress();
  console.log("Tracker deployed to:", contractAddress);

  // Save deployment data to deployments.json
  const deploymentsPath = path.join(__dirname, "..", "deployments.json");
  let deployments = [];

  // Load existing deployments if file exists
  if (fs.existsSync(deploymentsPath)) {
    const data = fs.readFileSync(deploymentsPath, "utf8");
    deployments = JSON.parse(data);
  }

  // Add new deployment
  const deployment = {
    network: hre.network.name,
    contractAddress: contractAddress,
    deployer: deployer.address,
    timestamp: new Date().toISOString(),
    chainId: hre.network.config.chainId || 1337,
    blockNumber: await hre.ethers.provider.getBlockNumber()
  };

  deployments.push(deployment);

  // Save to file
  fs.writeFileSync(deploymentsPath, JSON.stringify(deployments, null, 2));
  console.log("Deployment data saved to:", deploymentsPath);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });


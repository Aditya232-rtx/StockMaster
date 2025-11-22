require('dotenv').config();
const { ethers } = require('ethers');
const fetch = require('node-fetch');
const TrackerAbi = require('../frontend/src/TrackerAbi.json');

const RPC = process.env.RPC_URL;
const CONTRACT = process.env.CONTRACT_ADDRESS;
const WEBHOOK = process.env.WEBHOOK_URL;

if(!RPC || !CONTRACT) {
  console.error("Set RPC_URL and CONTRACT_ADDRESS in .env");
  process.exit(1);
}

const provider = new ethers.providers.JsonRpcProvider(RPC);
const contract = new ethers.Contract(CONTRACT, TrackerAbi, provider);

console.log("Listening for ItemUpdated events on", CONTRACT);

contract.on("ItemUpdated", (itemId, actor, status, timestamp, note, event) => {
  const payload = {
    itemId, actor, status: status.toNumber ? status.toNumber() : status, timestamp: Number(timestamp), note
  };
  console.log("ItemUpdated:", payload);

  if(WEBHOOK){
    fetch(WEBHOOK, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({text: 'ItemUpdated', payload})
    }).catch(err=>console.error("Webhook error", err));
  }
});

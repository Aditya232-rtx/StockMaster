/*
Simple QR code generator with hash generation.
Usage: node generate.js ITEM001
Outputs: qr_ITEM001.png in this folder.
Optional: set UPLOAD_IPFS=1 in env to upload to IPFS (via Infura) and print CID.
Set IPFS_PROJECT_ID and IPFS_PROJECT_SECRET in env for Infura IPFS.
*/
const QRCode = require("qrcode");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

async function uploadToIpfs(filePath) {
  try {
    const { create } = require('ipfs-http-client');
    const projectId = process.env.IPFS_PROJECT_ID;
    const projectSecret = process.env.IPFS_PROJECT_SECRET;
    let auth = undefined;
    if (projectId && projectSecret) {
      auth = 'Basic ' + Buffer.from(projectId + ':' + projectSecret).toString('base64');
    }
    const client = create({
      url: 'https://ipfs.infura.io:5001/api/v0',
      headers: auth ? { authorization: auth } : undefined
    });
    const data = fs.readFileSync(filePath);
    const added = await client.add(data);
    return added.cid.toString();
  } catch (err) {
    console.error("IPFS upload failed:", err.message || err);
    return null;
  }
}

function generateHash(itemId, metadata) {
  // Generate SHA256 hash from itemId and metadata
  const hashInput = JSON.stringify({ itemId, ...metadata });
  return crypto.createHash('sha256').update(hashInput).digest('hex');
}

function saveItemData(itemId, itemHash, metadata) {
  const itemsPath = path.join(__dirname, "items.json");
  let items = [];

  // Load existing items if file exists
  if (fs.existsSync(itemsPath)) {
    const data = fs.readFileSync(itemsPath, "utf8");
    items = JSON.parse(data);
  }

  // Add new item
  const item = {
    itemId,
    itemHash,
    metadata,
    createdAt: new Date().toISOString()
  };

  items.push(item);

  // Save to file
  fs.writeFileSync(itemsPath, JSON.stringify(items, null, 2));
  console.log("Item data saved to:", itemsPath);
}

async function generateQR(itemId, itemDetails = {}) {
  // Default values if not provided
  const defaultDetails = {
    itemName: `Item ${itemId}`,
    quantity: 1,
    pricePerItem: 0,
    dispatchTime: new Date().toISOString(),
    dispatchPlace: 'Unknown Location',
    ...itemDetails
  };

  const metadata = {
    itemName: defaultDetails.itemName,
    quantity: defaultDetails.quantity,
    pricePerItem: defaultDetails.pricePerItem,
    dispatchTime: defaultDetails.dispatchTime,
    dispatchPlace: defaultDetails.dispatchPlace
  };

  // Generate hash for this item
  const itemHash = generateHash(itemId, metadata);

  const qrData = {
    itemId,
    itemHash,
    ...metadata,
    createdAt: new Date().toISOString()
  };

  // Generate a clean, readable text version for the QR code
  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: true
    });
  };

  const qrText = `
  🛒 ITEM DETAILS
  ──────────────
  🔹 ID: ${itemId}
  🔹 Product: ${qrData.itemName}
  🔹 Quantity: ${qrData.quantity}
  🔹 Price: $${qrData.pricePerItem.toFixed(2)} each
  🔹 Total: $${(qrData.quantity * qrData.pricePerItem).toFixed(2)}
  
  🚚 SHIPPING INFORMATION
  ─────────────────────
  📅 Dispatch Date: ${formatDate(qrData.dispatchTime)}
  📍 Location: ${qrData.dispatchPlace}
  
  🔐 SECURITY
  ──────────
  Hash: ${itemHash.substring(0, 16)}...
  
  📝 ORDER SUMMARY
  ──────────────
  ${qrData.quantity} × ${qrData.itemName}
  @ $${qrData.pricePerItem.toFixed(2)} each
  
  💰 TOTAL: $${(qrData.quantity * qrData.pricePerItem).toFixed(2)}
  
  
  ⏰ Generated on: ${formatDate(qrData.createdAt)}
  `;

  const fileName = path.join(__dirname, `qr_${itemId}.png`);

  // Generate QR code with error correction for better reliability
  await QRCode.toFile(fileName, JSON.stringify(qrData), {
    width: 400,
    margin: 2,
    errorCorrectionLevel: 'H',
    color: {
      dark: '#1a1a1a',  // Darker color for better contrast
      light: '#ffffff'  // White background
    }
  });

  console.log("QR generated:", fileName);

  // Save item data to items.json
  saveItemData(itemId, itemHash, metadata);

  if (process.env.UPLOAD_IPFS === '1') {
    const cid = await uploadToIpfs(fileName);
    if (cid) {
      console.log("Uploaded to IPFS, CID:", cid);
      console.log("IPFS Gateway URL:", `https://ipfs.io/ipfs/${cid}`);
    }
  }

  return { fileName, data: qrData };
}

async function main() {
  const itemId = process.argv[2];
  if (!itemId) {
    console.log('Usage: node generate.js ITEM001 [itemName] [quantity] [price] [dispatchTime] [dispatchPlace]');
    console.log('Example: node generate.js ITEM001 "Apple iPhone 13" 1 999.99 "2023-12-01T10:00:00Z" "San Francisco, CA"');
    process.exit(0);
  }

  // Parse command line arguments
  const itemDetails = {
    itemName: process.argv[3] || `Item ${itemId}`,
    quantity: Number(process.argv[4]) || 1,
    pricePerItem: Number(process.argv[5]) || 0,
    dispatchTime: process.argv[6] || new Date().toISOString(),
    dispatchPlace: process.argv[7] || 'Unknown Location'
  };
  const { fileName, data } = await generateQR(itemId, itemDetails);

  // Print summary
  console.log("\n=== QR Code Details ===");
  console.log(`Item ID: ${data.itemId}`);
  console.log(`Item Hash: ${data.itemHash}`);
  console.log(`Item Name: ${data.itemName}`);
  console.log(`Quantity: ${data.quantity}`);
  console.log(`Price per Item: $${data.pricePerItem.toFixed(2)}`);
  console.log(`Total: $${(data.quantity * data.pricePerItem).toFixed(2)}`);
  console.log(`Dispatch: ${new Date(data.dispatchTime).toLocaleString()} from ${data.dispatchPlace}`);
  console.log(`QR Code saved to: ${fileName}`);
}

main().catch(err => { console.error(err); process.exit(1); });


/*
Bulk QR generator.
Usage: node bulk.js PREFIX COUNT
Example: node bulk.js ITEM 100
Outputs qr_ITEM0001.png ... qr_ITEM0100.png
Set UPLOAD_IPFS=1 to upload each generated PNG to IPFS (requires IPFS credentials in env).
*/
const QRCode = require("qrcode");
const fs = require("fs");
const path = require("path");

async function uploadToIpfs(filePath){
  try {
    const { create } = require('ipfs-http-client');
    const projectId = process.env.IPFS_PROJECT_ID;
    const projectSecret = process.env.IPFS_PROJECT_SECRET;
    let auth = undefined;
    if(projectId && projectSecret){
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

async function generateOne(itemId, index) {
  // Generate some sample data for bulk generation
  const itemDetails = {
    itemName: `Product ${String.fromCharCode(65 + (index % 26))}${Math.floor(index / 26) + 1}`,
    quantity: Math.floor(Math.random() * 10) + 1, // Random quantity 1-10
    pricePerItem: parseFloat((Math.random() * 1000).toFixed(2)), // Random price up to 1000
    dispatchedAt: new Date(Date.now() + (index * 24 * 60 * 60 * 1000)).toISOString(), // Staggered dispatch dates
    dispatchPlace: ['New York', 'San Francisco', 'Chicago', 'Miami', 'Seattle'][index % 5] // Rotate through locations
  };

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

  const qrData = {
    itemId,
    itemName: itemDetails.itemName,
    quantity: itemDetails.quantity,
    pricePerItem: itemDetails.pricePerItem,
    dispatchedAt: itemDetails.dispatchedAt,
    dispatchPlace: itemDetails.dispatchPlace
  };

  // Generate both JSON and human-readable text
  const qrText = JSON.stringify(qrData, null, 2) + "\n\n" + `
  🛒 ITEM DETAILS
  ──────────────
  🔹 ID: ${itemId}
  🔹 Product: ${itemDetails.itemName}
  🔹 Quantity: ${itemDetails.quantity}
  🔹 Price: $${itemDetails.pricePerItem.toFixed(2)} each
  🔹 Total: $${(itemDetails.quantity * itemDetails.pricePerItem).toFixed(2)}
  
  🚚 SHIPPING INFORMATION
  ─────────────────────
  📅 Dispatched At: ${formatDate(itemDetails.dispatchedAt)}
  📍 Location: ${itemDetails.dispatchPlace}
  
  
  📝 ORDER SUMMARY
  ──────────────
  ${itemDetails.quantity} × ${itemDetails.itemName}
  @ $${itemDetails.pricePerItem.toFixed(2)} each
  
  💰 TOTAL: $${(itemDetails.quantity * itemDetails.pricePerItem).toFixed(2)}
  
  
  ⏰ Generated on: ${formatDate(new Date().toISOString())}
  `;

  const fileName = path.join(__dirname, `qr_${itemId}.png`);
  
  await QRCode.toFile(fileName, qrText, { 
    width: 400, 
    margin: 1, 
    errorCorrectionLevel: 'H',
    color: {
      dark: '#1a1a1a',
      light: '#ffffff'
    }
  });
  
  return { 
    fileName, 
    details: { itemId, ...itemDetails }
  };
}

async function main(){
  const prefix = process.argv[2] || "ITEM";
  const count = Math.min(Number(process.argv[3]) || 10, 100); // Limit to 100 max for safety
  
  console.log(`Generating ${count} QR codes with prefix '${prefix}'...\n`);
  
  const results = [];
  
  for(let i=1; i<=count; i++){
    const id = prefix + String(i).padStart(4,'0');
    const { fileName, details } = await generateOne(id, i);
    
    // Log progress
    console.log(`[${i}/${count}] Generated: ${fileName}`);
    console.log(`   ${details.quantity}x ${details.itemName.padEnd(15)} @ $${details.pricePerItem.toFixed(2).padStart(8)} each`);
    console.log(`   Dispatch: ${new Date(details.dispatchTime).toLocaleDateString()} from ${details.dispatchPlace}\n`);
    
    if(process.env.UPLOAD_IPFS === '1'){
      const cid = await uploadToIpfs(fileName);
      if(cid) {
        console.log(`   Uploaded to IPFS: https://ipfs.io/ipfs/${cid}\n`);
        results.push({ ...details, ipfsCid: cid });
      }
    } else {
      results.push(details);
    }
  }
  
  // Generate a summary report
  console.log("\n=== Batch Generation Complete ===");
  console.log(`Total QR Codes Generated: ${results.length}`);
  
  if(results.length > 0) {
    const totalValue = results.reduce((sum, item) => 
      sum + (item.quantity * item.pricePerItem), 0);
      
    console.log(`Total Value: $${totalValue.toFixed(2)}`);
    
    // Save results to a JSON file
    const reportFile = path.join(__dirname, `batch_${prefix}_${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
    fs.writeFileSync(reportFile, JSON.stringify(results, null, 2));
    console.log(`\nBatch report saved to: ${reportFile}`);
  }
}

// Handle errors and cleanup
process.on('unhandledRejection', error => {
  console.error('Unhandled Promise Rejection:', error);
  process.exit(1);
});

process.on('SIGINT', () => {
  console.log('\nOperation cancelled by user');
  process.exit(0);
});

main().catch(err => { 
  console.error('Error during batch generation:', err); 
  process.exit(1); 
});

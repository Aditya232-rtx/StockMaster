import React, { useState } from 'react';
import QRCode from 'qrcode';
import { ethers } from 'ethers';

export default function GenerateQR() {
  const [itemId, setItemId] = useState('');
  const [qrDataUrl, setQrDataUrl] = useState('');
  const [itemHash, setItemHash] = useState('');

  function generateHash(itemId, metadata = {}) {
    const hashInput = JSON.stringify({ itemId, ...metadata });
    return ethers.utils.id(hashInput);
  }

  async function generate() {
    if (!itemId) return alert('Enter item id');

    const metadata = {
      createdAt: new Date().toISOString()
    };

    const hash = generateHash(itemId, metadata);
    setItemHash(hash);

    const payload = JSON.stringify({
      itemId,
      itemHash: hash,
      ...metadata
    });

    const url = await QRCode.toDataURL(payload, { width: 400 });
    setQrDataUrl(url);
  }

  async function download() {
    if (!qrDataUrl) return;
    const a = document.createElement('a');
    a.href = qrDataUrl;
    a.download = `qr_${itemId}.png`;
    a.click();
  }

  return (
    <div className="bg-card-bg rounded-xl shadow-md p-6 border border-accent">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Generate QR Code</h2>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Transaction ID: <span className="text-blue-600">{itemId || "Not Set"}</span>
          </label>
          <input
            type="text"
            placeholder="Enter Item ID"
            value={itemId}
            onChange={e => setItemId(e.target.value)}
            className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        <button
          onClick={generate}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition duration-200"
        >
          Generate
        </button>

        {itemHash && (
          <div className="bg-accent bg-opacity-30 rounded-lg p-3 border border-accent">
            <p className="text-xs font-medium text-gray-700 mb-1">Generated Hash:</p>
            <p className="text-xs text-gray-600 break-all">{itemHash.substring(0, 40)}...</p>
          </div>
        )}

        {qrDataUrl && (
          <div className="space-y-3">
            <div className="bg-white rounded-lg p-4 flex justify-center border border-gray-200">
              <img src={qrDataUrl} alt="qr" className="max-w-xs" />
            </div>

            <div className="flex gap-3">
              <button
                onClick={download}
                className="flex-1 border border-gray-300 hover:bg-gray-50 text-gray-700 font-medium py-2 px-4 rounded-lg transition duration-200"
              >
                Download PNG
              </button>
              <button
                onClick={() => alert('IPFS upload requires server-side implementation')}
                className="flex-1 border border-gray-300 hover:bg-gray-50 text-gray-700 font-medium py-2 px-4 rounded-lg transition duration-200"
              >
                Upload to IPFS
              </button>
            </div>
            <p className="text-xs text-gray-500 text-center">(server-side recommended)</p>
          </div>
        )}
      </div>
    </div>
  );
}

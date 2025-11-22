import React, { useState } from 'react';
import QRCode from 'qrcode';
import { ethers } from 'ethers';
import { QrCode, Download, Upload as UploadIcon, RefreshCw } from 'lucide-react';

export default function GenerateQR() {
    const [itemId, setItemId] = useState('193786'); // Default as requested
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
        <div className="bg-[#F8F7FE] rounded-2xl shadow-sm p-8 border border-white/60 h-full">
            <h2 className="text-xl font-bold text-black mb-6 flex items-center gap-3">
                <div className="p-2 bg-white rounded-lg shadow-sm">
                    <QrCode className="w-5 h-5 text-black" />
                </div>
                QR Code Generator
            </h2>

            <div className="space-y-8">
                <div>
                    <div className="flex items-center justify-between mb-3">
                        <label className="block text-sm font-bold text-gray-800">Transaction ID</label>
                        <span className="text-xs font-medium text-gray-500 bg-white px-2 py-1 rounded-md border border-gray-100">Required</span>
                    </div>
                    <div className="relative">
                        <input
                            type="text"
                            value={itemId}
                            onChange={e => setItemId(e.target.value)}
                            className="w-full border border-gray-200 rounded-xl p-4 bg-white font-mono text-base text-black focus:ring-2 focus:ring-black focus:border-transparent outline-none transition-all shadow-sm"
                            placeholder="Enter transaction ID"
                        />
                    </div>
                </div>

                <button
                    onClick={generate}
                    className="w-full bg-black hover:bg-gray-800 text-white font-bold py-4 px-6 rounded-xl transition-all shadow-md hover:shadow-lg flex items-center justify-center gap-3"
                >
                    <RefreshCw className={`w-5 h-5 ${qrDataUrl ? 'animate-spin' : ''}`} />
                    <span>{qrDataUrl ? 'Regenerate' : 'Generate QR Code'}</span>
                </button>

                {qrDataUrl ? (
                    <div className="space-y-6 pt-2">
                        <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm flex flex-col items-center justify-center">
                            <img
                                src={qrDataUrl}
                                alt="QR Code"
                                className="w-56 h-56 object-contain"
                            />
                            <p className="mt-4 text-sm font-medium text-gray-500 text-center bg-gray-50 px-4 py-2 rounded-lg w-full">
                                Scan to verify transaction
                            </p>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <button
                                onClick={download}
                                className="bg-white hover:bg-gray-50 text-black border border-gray-200 font-bold py-3 px-4 rounded-xl transition-all flex items-center justify-center gap-2 shadow-sm"
                            >
                                <Download className="w-4 h-4" />
                                <span>Download</span>
                            </button>
                            <button
                                onClick={() => alert('IPFS upload requires server-side implementation')}
                                className="bg-white hover:bg-gray-50 text-black border border-gray-200 font-bold py-3 px-4 rounded-xl transition-all flex items-center justify-center gap-2 shadow-sm"
                            >
                                <UploadIcon className="w-4 h-4" />
                                <span>IPFS Upload</span>
                            </button>
                        </div>

                        <div className="text-center border-t border-gray-100 pt-4">
                            <p className="text-xs text-gray-400 font-medium">
                                Secure Blockchain Verification
                            </p>
                        </div>
                    </div>
                ) : (
                    <div className="bg-white rounded-2xl p-10 text-center border-2 border-dashed border-gray-200">
                        <div className="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center mx-auto mb-4">
                            <QrCode className="w-8 h-8 text-gray-300" />
                        </div>
                        <p className="text-sm font-medium text-gray-400">
                            Generated QR code will appear here
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}

import React, { useState, useRef, useEffect } from "react";
import { ethers } from "ethers";
import { Html5Qrcode } from "html5-qrcode";
import { Wallet, QrCode, Scan, Upload, Download, CheckCircle, AlertCircle } from 'lucide-react';
import TrackerAbi from "./TrackerAbi.json";
import GenerateQR from './GenerateQR';
import './styles.css';

// Try to load contract address from deployments.json
const TRACKER_ADDRESS = "0x5FC8d32690cc91D4c39d9d3abcBD16989F875707"; // Fallback

export default function App() {
    const [provider, setProvider] = useState(null);
    const [signer, setSigner] = useState(null);
    const [contract, setContract] = useState(null);
    const [itemId, setItemId] = useState("S6743578"); // Default as requested
    const [note, setNote] = useState("");
    const [status, setStatus] = useState(1);
    const [logs, setLogs] = useState([]);
    const [itemHash, setItemHash] = useState("");
    const [contractAddress, setContractAddress] = useState(TRACKER_ADDRESS);
    const qrRef = useRef(null);
    const scannerRef = useRef(null);
    const [isScanning, setIsScanning] = useState(false);

    // Load contract address from deployments.json on mount
    useEffect(() => {
        async function loadDeploymentData() {
            try {
                const response = await fetch('/deployments.json');
                if (response.ok) {
                    const deployments = await response.json();
                    const latestDeployment = deployments
                        .filter(d => d.network === 'localhost')
                        .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0];

                    if (latestDeployment) {
                        setContractAddress(latestDeployment.contractAddress);
                    }
                }
            } catch (err) {
                console.log("Could not load deployments.json, using fallback address");
            }
        }
        loadDeploymentData();
    }, []);

    function generateHash(itemId, metadata = {}) {
        const hashInput = JSON.stringify({ itemId, ...metadata });
        return ethers.utils.id(hashInput);
    }

    async function connect() {
        if (window.ethereum) {
            const p = new ethers.providers.Web3Provider(window.ethereum);
            await p.send("eth_requestAccounts", []);
            const s = p.getSigner();
            setProvider(p);
            setSigner(s);
            const c = new ethers.Contract(contractAddress, TrackerAbi, s);
            setContract(c);
            alert("Connected");
        } else {
            alert("Install MetaMask");
        }
    }

    async function sendUpdate() {
        if (!contract) return alert("Connect first");
        try {
            const tx = await contract.updateItem(itemId, status, note);
            await tx.wait();
            alert("Update recorded. Tx: " + tx.hash);
            loadHistory();
        } catch (err) {
            alert("Update failed: " + err.message);
        }
    }

    async function loadHistory() {
        if (!contract) return;
        try {
            const len = await contract.getUpdatesLength(itemId);
            const l = [];
            for (let i = 0; i < Number(len); i++) {
                const u = await contract.getUpdateByIndex(itemId, i);
                l.push({
                    actor: u.actor,
                    status: u.status.toNumber(),
                    note: u.note,
                    timestamp: new Date(u.timestamp.toNumber() * 1000).toLocaleString(),
                    itemHash: u.itemHash
                });
            }
            setLogs(l);

            if (len > 0) {
                const hash = await contract.getItemHash(itemId);
                setItemHash(hash);
            }
        } catch (err) {
            console.error("Error loading history:", err);
        }
    }

    function startScanner() {
        if (scannerRef.current) return;
        const html5QrCode = new Html5Qrcode("qr-reader");
        scannerRef.current = html5QrCode;
        setIsScanning(true);
        html5QrCode.start(
            { facingMode: "environment" },
            { fps: 10, qrbox: 250 },
            (decodedText, decodedResult) => {
                try {
                    const data = JSON.parse(decodedText);
                    setItemId(data.itemId);
                    if (data.itemHash) {
                        setItemHash(data.itemHash);
                    }
                } catch (e) {
                    setItemId(decodedText);
                }
                html5QrCode.stop().then(() => {
                    scannerRef.current = null;
                    setIsScanning(false);
                }).catch(() => { });
            },
            (errorMessage) => { }
        ).catch(err => {
            alert("Unable to start camera. " + err);
            setIsScanning(false);
        });
    }

    function stopScanner() {
        if (scannerRef.current) {
            scannerRef.current.stop();
            scannerRef.current = null;
            setIsScanning(false);
        }
    }

    return (
        <div className="min-h-screen bg-[#EDF6F7] p-4 sm:p-8 font-sans text-black">
            {/* Header */}
            <header className="mb-8 max-w-7xl mx-auto">
                <h1 className="text-3xl font-bold text-black tracking-tight">Blockchain Tracker</h1>
                <div className="mt-3 flex items-center text-sm text-gray-700 bg-[#F8F7FE] px-4 py-2 rounded-xl w-fit shadow-sm border border-white/50">
                    <span className="font-mono text-xs">
                        Contract: {contractAddress}
                    </span>
                </div>
            </header>

            {/* Main Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-7xl mx-auto">
                {/* LEFT COLUMN */}
                <div className="space-y-8">
                    {/* Wallet Connection Card */}
                    <div className="bg-[#F8F7FE] rounded-2xl shadow-sm p-8 border border-white/60">
                        <h2 className="text-xl font-bold text-black mb-6 flex items-center gap-3">
                            <div className="p-2 bg-white rounded-lg shadow-sm">
                                <Wallet className="w-5 h-5 text-black" />
                            </div>
                            Wallet Connection
                        </h2>

                        <button
                            onClick={connect}
                            className={`w-full py-4 px-6 rounded-xl font-bold mb-8 transition-all flex items-center justify-center gap-3 shadow-sm hover:shadow-md ${provider
                                    ? 'bg-green-500 hover:bg-green-600 text-white'
                                    : 'bg-black hover:bg-gray-800 text-white'
                                }`}
                        >
                            {provider ? (
                                <>
                                    <CheckCircle className="w-5 h-5" />
                                    <span>Connected to MetaMask</span>
                                </>
                            ) : (
                                <>
                                    <Wallet className="w-5 h-5" />
                                    <span>Connect MetaMask</span>
                                </>
                            )}
                        </button>

                        {/* Scanner Section */}
                        <div className="space-y-6">
                            <div
                                id="qr-reader"
                                className="rounded-2xl overflow-hidden bg-white min-h-[240px] flex items-center justify-center border-2 border-dashed border-gray-200"
                            >
                                {!isScanning ? (
                                    <div className="text-center p-6">
                                        <div className="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center mx-auto mb-4">
                                            <Scan className="w-8 h-8 text-gray-400" />
                                        </div>
                                        <p className="text-sm font-medium text-gray-500">Camera Preview Area</p>
                                    </div>
                                ) : null}
                            </div>

                            <div className="flex gap-4">
                                <button
                                    onClick={startScanner}
                                    disabled={isScanning}
                                    className={`flex-1 py-3 px-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all ${isScanning
                                            ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                                            : 'bg-white border border-gray-200 text-black hover:bg-gray-50 hover:border-gray-300 shadow-sm'
                                        }`}
                                >
                                    <Scan className="w-4 h-4" />
                                    <span>Start Scanner</span>
                                </button>
                                <button
                                    onClick={stopScanner}
                                    disabled={!isScanning}
                                    className={`flex-1 py-3 px-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all ${!isScanning
                                            ? 'bg-gray-50 text-gray-300 cursor-not-allowed border border-gray-100'
                                            : 'bg-white border border-gray-200 text-red-600 hover:bg-red-50 hover:border-red-200 shadow-sm'
                                        }`}
                                >
                                    <span>Stop Scanner</span>
                                </button>
                            </div>

                            {/* Status Display */}
                            <div className="p-4 bg-white rounded-xl border border-gray-100 shadow-sm">
                                <div className="flex items-center gap-3 text-sm">
                                    <div className="relative flex h-3 w-3">
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                                        <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                                    </div>
                                    <span className="font-bold text-gray-900">System Status:</span>
                                    <span className="text-gray-600">Ready to scan</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="grid grid-cols-2 gap-4">
                        <button
                            onClick={sendUpdate}
                            className="bg-black hover:bg-gray-800 text-white font-bold py-4 px-6 rounded-xl transition-all shadow-md hover:shadow-lg flex items-center justify-center gap-2"
                        >
                            <Upload className="w-5 h-5" />
                            <span>Send Update</span>
                        </button>
                        <button
                            onClick={loadHistory}
                            className="bg-white hover:bg-gray-50 text-black border border-gray-200 font-bold py-4 px-6 rounded-xl transition-all shadow-sm hover:shadow-md flex items-center justify-center gap-2"
                        >
                            <Download className="w-5 h-5" />
                            <span>Load History</span>
                        </button>
                    </div>
                </div>

                {/* RIGHT COLUMN */}
                <div className="space-y-6">
                    {/* Generate QR Code Card */}
                    <GenerateQR />
                </div>
            </div>
        </div>
    )
}

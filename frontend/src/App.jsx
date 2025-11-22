import React, { useState, useRef, useEffect } from "react";
import { ethers } from "ethers";
import { Html5Qrcode } from "html5-qrcode";
import TrackerAbi from "./TrackerAbi.json";
import GenerateQR from './GenerateQR';
import './styles.css';

// Try to load contract address from deployments.json
let TRACKER_ADDRESS = "0x5FC8d32690cc91D4c39d9d3abcBD16989F875707"; // Fallback

export default function App() {
  const [provider, setProvider] = useState(null);
  const [signer, setSigner] = useState(null);
  const [contract, setContract] = useState(null);
  const [itemId, setItemId] = useState("");
  const [note, setNote] = useState("");
  const [status, setStatus] = useState(1);
  const [logs, setLogs] = useState([]);
  const [itemHash, setItemHash] = useState("");
  const [contractAddress, setContractAddress] = useState(TRACKER_ADDRESS);
  const qrRef = useRef(null);
  const scannerRef = useRef(null);

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
            console.log("Loaded contract address from deployments.json:", latestDeployment.contractAddress);
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

  async function registerItem() {
    if (!contract) return alert("Connect first");
    if (!itemId) return alert("Enter item ID");

    const hash = generateHash(itemId, {});
    setItemHash(hash);

    try {
      const tx = await contract.registerItem(itemId, hash);
      await tx.wait();
      alert(`Item registered! Hash: ${hash.substring(0, 16)}...`);
    } catch (err) {
      alert("Registration failed: " + err.message);
    }
  }

  async function sendUpdate() {
    if (!contract) return alert("Connect first");
    const tx = await contract.updateItem(itemId, status, note);
    await tx.wait();
    alert("Update recorded. Tx: " + tx.hash);
    loadHistory();
  }

  async function loadHistory() {
    if (!contract) return;
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
  }

  function startScanner() {
    if (scannerRef.current) return;
    const html5QrCode = new Html5Qrcode("qr-reader");
    scannerRef.current = html5QrCode;
    html5QrCode.start(
      { facingMode: "environment" },
      { fps: 10, qrbox: 250 },
      (decodedText, decodedResult) => {
        try {
          const data = JSON.parse(decodedText);
          setItemId(data.itemId);
          if (data.itemHash) {
            setItemHash(data.itemHash);
            console.log("Scanned item with hash:", data.itemHash);
          }
        } catch (e) {
          setItemId(decodedText);
        }
        html5QrCode.stop().then(() => { scannerRef.current = null; }).catch(() => { });
      },
      (errorMessage) => { }
    ).catch(err => {
      alert("Unable to start camera. " + err);
    });
  }

  function stopScanner() {
    if (scannerRef.current) {
      scannerRef.current.stop();
      scannerRef.current = null;
    }
  }

  return (
    <div className="min-h-screen bg-page-bg">
      {/* Header */}
      <header className="bg-card-bg shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-3xl font-semibold text-gray-900">Blockchain Tracker</h1>
          <p className="text-sm text-gray-500 mt-1">Contract: {contractAddress.substring(0, 10)}...{contractAddress.substring(38)}</p>
        </div>
      </header>

      {/* Main Container */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* Column 1 */}
          <div className="space-y-6">

            {/* Card 1: Wallet Connection */}
            <div className="bg-card-bg rounded-xl shadow-md p-6 border border-accent">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Wallet Connection</h2>
              <button
                onClick={connect}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition duration-200 mb-4"
              >
                Connect MetaMask
              </button>

              <div className="mt-6">
                <h3 className="text-sm font-medium text-gray-700 mb-3">QR Scanner</h3>
                <div id="qr-reader" ref={qrRef} className="mb-4 rounded-lg overflow-hidden"></div>
                <div className="flex gap-3">
                  <button
                    onClick={startScanner}
                    className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition duration-200"
                  >
                    Start Scanner
                  </button>
                  <button
                    onClick={stopScanner}
                    className="flex-1 border border-gray-300 hover:bg-gray-50 text-gray-700 font-medium py-2 px-4 rounded-lg transition duration-200"
                  >
                    Stop Scanner
                  </button>
                </div>
              </div>
            </div>

            {/* Card 2: Student Record */}
            <div className="bg-card-bg rounded-xl shadow-md p-6 border border-accent">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">
                Item Record: <span className="text-blue-600">{itemId || "Not Set"}</span>
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Item ID (from QR)</label>
                  <input
                    type="text"
                    placeholder="Enter or scan item ID"
                    value={itemId}
                    onChange={e => setItemId(e.target.value)}
                    className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  {itemHash && (
                    <p className="text-xs text-gray-500 mt-1">
                      Hash: {itemHash.substring(0, 20)}...
                    </p>
                  )}
                </div>

                <button
                  onClick={registerItem}
                  className="w-full bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded-lg transition duration-200"
                >
                  Register Item
                </button>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Checkpoint</label>
                  <select
                    value={status}
                    onChange={e => setStatus(Number(e.target.value))}
                    className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value={1}>Packed</option>
                    <option value={2}>Dispatched</option>
                    <option value={3}>InTransit</option>
                    <option value={4}>Checkpoint</option>
                    <option value={5}>Delivered</option>
                    <option value={6}>Received</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Note</label>
                  <input
                    type="text"
                    placeholder="Enter note"
                    value={note}
                    onChange={e => setNote(e.target.value)}
                    className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div className="flex gap-3 pt-2">
                  <button
                    onClick={sendUpdate}
                    className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition duration-200"
                  >
                    Send Update
                  </button>
                  <button
                    onClick={loadHistory}
                    className="flex-1 border border-gray-300 hover:bg-gray-50 text-gray-700 font-medium py-3 px-4 rounded-lg transition duration-200"
                  >
                    Load History
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Column 2 */}
          <div className="space-y-6">

            {/* Card 3: Generate QR Code */}
            <GenerateQR />

            {/* Card 4: History */}
            <div className="bg-card-bg rounded-xl shadow-md p-6 border border-accent">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">History</h2>

              {logs.length === 0 ? (
                <div className="bg-gray-50 rounded-lg p-8 text-center">
                  <p className="text-gray-500">No history yet. Load history to see updates.</p>
                </div>
              ) : (
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {logs.map((l, idx) => (
                    <div key={idx} className="bg-accent bg-opacity-30 rounded-lg p-4 border border-accent">
                      <div className="flex justify-between items-start mb-2">
                        <span className="text-sm font-semibold text-gray-900">{l.timestamp}</span>
                        <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
                          Status {l.status}
                        </span>
                      </div>
                      <p className="text-sm text-gray-700 mb-1">{l.note}</p>
                      <p className="text-xs text-gray-500">By: {l.actor.substring(0, 10)}...{l.actor.substring(38)}</p>
                      {l.itemHash && l.itemHash !== ethers.constants.HashZero && (
                        <p className="text-xs text-gray-400 mt-1">Hash: {l.itemHash.substring(0, 16)}...</p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Chart Visualization Placeholder */}
            <div className="bg-card-bg rounded-xl shadow-md p-6 border border-accent">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Visualization</h2>
              <div className="bg-gray-100 rounded-lg h-64 flex items-center justify-center">
                <p className="text-gray-500 text-center">Chart visualization will appear here...</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

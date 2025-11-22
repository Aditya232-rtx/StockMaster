// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/access/Ownable.sol";

contract Tracker is Ownable {
    enum Status { None, Packed, Dispatched, InTransit, Checkpoint, Delivered, Received }

    struct Update {
        address actor;
        Status status;
        string note;
        uint256 timestamp;
        bytes32 itemHash;
    }

    mapping(string => Update[]) private updates;
    mapping(address => bool) public authorized;
    mapping(bytes32 => bool) public registeredItems;
    mapping(string => bytes32) public itemIdToHash;

    event ItemUpdated(string indexed itemId, address indexed actor, Status status, uint256 timestamp, string note);
    event ItemRegistered(string indexed itemId, bytes32 indexed itemHash, uint256 timestamp);

    modifier onlyAuthorized() {
        require(authorized[msg.sender] || owner() == msg.sender, "Not authorized");
        _;
    }

    function setAuthorized(address who, bool allowed) external onlyOwner {
        authorized[who] = allowed;
    }

    function registerItem(string calldata itemId, bytes32 itemHash) external onlyAuthorized {
        require(!registeredItems[itemHash], "Item hash already registered");
        require(itemIdToHash[itemId] == bytes32(0), "Item ID already registered");
        
        registeredItems[itemHash] = true;
        itemIdToHash[itemId] = itemHash;
        
        emit ItemRegistered(itemId, itemHash, block.timestamp);
    }

    function updateItem(string calldata itemId, Status status, string calldata note) external onlyAuthorized {
        bytes32 storedHash = itemIdToHash[itemId];
        
        Update memory u = Update({
            actor: msg.sender,
            status: status,
            note: note,
            timestamp: block.timestamp,
            itemHash: storedHash
        });
        updates[itemId].push(u);
        emit ItemUpdated(itemId, msg.sender, status, block.timestamp, note);
    }

    function getLatest(string calldata itemId) external view returns (Update memory) {
        require(updates[itemId].length > 0, "No updates");
        return updates[itemId][updates[itemId].length - 1];
    }

    function getUpdatesLength(string calldata itemId) external view returns (uint256) {
        return updates[itemId].length;
    }

    function getUpdateByIndex(string calldata itemId, uint256 index) external view returns (Update memory) {
        require(index < updates[itemId].length, "Index out of bounds");
        return updates[itemId][index];
    }

    function getItemHash(string calldata itemId) external view returns (bytes32) {
        return itemIdToHash[itemId];
    }

    function isItemRegistered(bytes32 itemHash) external view returns (bool) {
        return registeredItems[itemHash];
    }
}

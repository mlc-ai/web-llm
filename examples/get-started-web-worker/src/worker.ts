import { WebWorkerMLCEngineHandler } from "@mlc-ai/web-llm";
import { RTCPeerConnection, RTCSessionDescription } from "wrtc";
import DHT from "bittorrent-dht";
import crypto from "crypto";

// Hookup an engine to a worker handler
const handler = new WebWorkerMLCEngineHandler();
self.onmessage = (msg: MessageEvent) => {
  handler.onmessage(msg);
};

// WebRTC and DHT integration
const dht = new DHT();
const peers = new Map<string, RTCPeerConnection>();

dht.listen(20000, () => {
  console.log("DHT listening on port 20000");
});

dht.on("peer", (peer, infoHash, from) => {
  console.log(`Found potential peer ${peer.host}:${peer.port} through DHT`);
  connectToPeer(peer.host, peer.port);
});

function connectToPeer(host: string, port: number) {
  const peerId = `${host}:${port}`;
  if (peers.has(peerId)) {
    return;
  }

  const peerConnection = new RTCPeerConnection();
  peers.set(peerId, peerConnection);

  peerConnection.onicecandidate = (event) => {
    if (event.candidate) {
      dht.announce(peerId, event.candidate);
    }
  };

  peerConnection.ondatachannel = (event) => {
    const dataChannel = event.channel;
    dataChannel.onmessage = (event) => {
      console.log(`Received message from ${peerId}: ${event.data}`);
      displayReceivedMessage(peerId, event.data);
    };
  };

  const dataChannel = peerConnection.createDataChannel("data");
  dataChannel.onopen = () => {
    console.log(`Data channel with ${peerId} is open`);
  };

  peerConnection.createOffer().then((offer) => {
    return peerConnection.setLocalDescription(offer);
  }).then(() => {
    dht.announce(peerId, peerConnection.localDescription);
  });
}

// Piece selection algorithm
function selectPiece(pieces: Array<{ index: number, rarity: number }>): number {
  pieces.sort((a, b) => a.rarity - b.rarity);
  return pieces[0].index;
}

// Function to send message to other nodes using WebRTC
function sendMessageToNode(peerId: string, message: string) {
  const peerConnection = peers.get(peerId);
  if (peerConnection) {
    const dataChannel = peerConnection.createDataChannel("data");
    dataChannel.onopen = () => {
      dataChannel.send(message);
      console.log(`Sent message to ${peerId}: ${message}`);
    };
  } else {
    console.log(`Peer ${peerId} not found`);
  }
}

// Function to display received messages from other nodes
function displayReceivedMessage(peerId: string, message: string) {
  console.log(`Message from ${peerId}: ${message}`);
}

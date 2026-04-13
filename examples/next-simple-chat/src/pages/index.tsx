import Head from "next/head";
import dynamic from "next/dynamic";
import { Inter } from "next/font/google";

const inter = Inter({ subsets: ["latin"] });
const ChatComponent = dynamic(() => import("~/utils/chat_component"), {
  ssr: false,
});

export default function Home() {
  return (
    <>
      <Head>
        <title>Example App</title>
        <meta
          name="description"
          content="Example app for web llm next compatibility"
        />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main
        className={`flex min-h-screen flex-col items-center justify-between p-24 ${inter.className}`}
      >
        <ChatComponent />
      </main>
    </>
  );
}

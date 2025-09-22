import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
JobExecutorType,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    WorkerPermissions,
    WorkerType,
    metrics,
    JobRequest,
    CloseEvent,
    utils
)
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from graph import Graph
import asyncio
from livekit.agents.utils.misc import shortuuid


logger = logging.getLogger("agent")

load_dotenv('../.env.local')


# Check if the LIVEKIT_API_KEY is set
livekit_api_key = os.getenv("LIVEKIT_API_KEY")

if livekit_api_key:
    logger.info("LIVEKIT_API_KEY is loaded.")
else:
    logger.info("LIVEKIT_API_KEY is not loaded.")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

async def _final_graph(conversation):
        if conversation:
            await _invoke_graph(" ".join(conversation))

async def invoke_graph(text):
    logger.info("\n\n\n\nInvoking graph with text...\n\n\n\n")
    result = await Graph.ainvoke({"raw_scribe": text})
    return result

async def _gen_insights(text):
    logger.info("Invoking graph for creating insights...")
    result = await Graph.ainvoke({"raw_scribe": text})
    return result['insights']

async def request_fnc(req: JobRequest):
    logger.info(f"Received job request: {req}")
    await req.accept(
        name="voice-assistant",
        identity=f"agent-{req.job.id}",
        attributes={"agent_type": "voice_assistant"}
    )
    logger.info("Job request accepted")

async def _invoke_graph(text: str) -> None:
    logger.info("Invoking graph with %d chars", len(text))
    try:
        result = await Graph.ainvoke({"raw_scribe": text})
        logger.info("Graph insights: %s", result.get("insights"))
    except Exception as e:
        logger.exception("Graph failed: %s", e)


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    conversation = []
    counter = 0    
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead:
    # session = AgentSession(
    #     # See all providers at https://docs.livekit.io/agents/integrations/realtime/
    #     llm=openai.realtime.RealtimeModel()
    # )

    # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
    # when it's detected, you may resume the agent's speech
    room = ctx.room
    @session.on("user_input_transcribed")
    def on_transcript(ev):
        nonlocal conversation, counter
        if not ev.is_final:
            return

        conversation.append(ev.transcript)
        counter += 1
        loop = asyncio.get_running_loop()

        # 1. Forward the user transcript
        # loop.create_task(
        #     room.local_participant.send_text(
        #         ev.transcript,
        #         topic="lk.transcription",
        #         attributes={"lk.transcribed_track_id": shortuuid()}
        #     )
        # )

        # 2. Every 10th sentence send interim insights
        if counter % 10 == 0:
            async def _send_insights():
                try:
                    text = " ".join(conversation)
                    result = await invoke_graph(text)
                    logger.info(result)

                    # Extract data from the graph result
                    insights_list = result.get('insights', [])
                    probing_questions = result.get('probing_questions', [])
                    chat_note = result.get('chat_note', '')
                    red_flags = result.get('red_flags', [])
                    
                    logger.info(
                        "Interim insights at %s: %s", counter, insights_list
                    )

                    # Generate payloads for each topic
                    payloads = {}
                    
                    # Insights topic
                    if isinstance(insights_list, list) and insights_list:
                        payloads['insights'] = "\n".join(insights_list)
                    else:
                        logger.warning("Insights is not a list: %s", type(insights_list))
                        payloads['insights'] = "No insights generated"

                    # Notes topic (chat note)
                    if chat_note and isinstance(chat_note, str):
                        payloads['notes'] = chat_note
                    else:
                        payloads['notes'] = "No chat note generated"

                    # Probing questions topic
                    if isinstance(probing_questions, list) and probing_questions:
                        payloads['probing'] = "\n".join(probing_questions)
                    else:
                        payloads['probing'] = "No probing questions generated"

                    # Red flags topic
                    if isinstance(red_flags, list) and red_flags:
                        payloads['redflags'] = "\n".join(red_flags)
                    else:
                        payloads['redflags'] = "No red flags identified"

                    # Send to all topics
                    topics = {
                        'insights': 'lk.insights',
                        'notes': 'lk.notes', 
                        'probing': 'lk.probing',
                        'redflags': 'lk.redflags'
                    }

                    for content_type, topic in topics.items():
                        payload = payloads[content_type]
                        track_id = shortuuid()
                        await room.local_participant.send_text(
                            payload,
                            topic=topic,
                            attributes={
                                "lk.transcribed_track_id": track_id,
                                "lk.content_type": content_type  # Additional attribute to identify content type
                            }
                        )
                        logger.info(f"Sent {content_type} to topic {topic} with payload length: {len(payload)}")

                except Exception as e:
                    logger.error("Failed to send interim insights: %s", e)
            loop.create_task(_send_insights())    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    #Calculate graph when the track is muted
    async def _on_track_muted(publication, *args):
        nonlocal conversation
        logger.info("Track muted, triggering graph invocation...")
        if not conversation:
            logger.info("Conversation is empty, nothing to analyze.")
            return

        text = " ".join(conversation)
        try:
            result = await invoke_graph(text)

            # Flatten the fields into one string
            payload = (
                f"Graph Insights: {', '.join(result.get('insights', []))}\n"
                f"Probing Questions: {'; '.join(result.get('probing_questions', []))}\n"
                f"Notes: {result.get('chat_note', ' ')}\n"
                f"Red Flags: {', '.join(result.get('red_flags', []))}"
                
            )
            

            await room.local_participant.send_text(
                payload,
                topic="lk.notes",
                attributes={"lk.transcribed_track_id": shortuuid()}
            )
            logger.info("Graph result sent to room successfully.")
        except Exception as e:
            logger.error("Failed to invoke graph or send result: %e", e)


    # register async callback correctly
    ctx.room.on("track_muted", lambda pub, *a: asyncio.create_task(_on_track_muted(pub, *a)))
    
    @session.on("close")
    def on_close(event: CloseEvent):
        #we can send all the data back to 
        logger.info("Session closed")
        logger.info(f"Conversation: {' '.join(conversation)}")
        result = asyncio.run(invoke_graph(" ".join(conversation)))
        logger.info(f"Final Graph Result: {result}")
        
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
        logger.info("Saving the data into database... (Mock function database not implimented yet)")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/integrations/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/integrations/avatar/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    
    try:
        await session.start(
            room=ctx.room,
            agent=Assistant(),
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
                ),
        )
        logger.info("Agent session started successfully")
    except Exception as e:
        logger.error(f"Failed to start agent session: {e}")



    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    opts = WorkerOptions(
        entrypoint_fnc=entrypoint,
        worker_type=WorkerType.ROOM,
        request_fnc=request_fnc,
        job_executor_type=JobExecutorType.THREAD,  # ← many rooms per process
        job_memory_warn_mb=1024,  # Warn at 1GB instead of 500MB
        job_memory_limit_mb=4096,  # Set limit at 4GB for safety
        initialize_process_timeout=500,
        permissions=WorkerPermissions(
            can_publish=True,
            can_subscribe=True,     
            can_publish_data=True,
            hidden=False
        )
    )
    cli.run_app(opts)
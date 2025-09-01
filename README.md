
# LiveKit Scribing Agent - Python


## Dev Setup

Clone the repository and install dependencies to a virtual environment:

```console
uv sync
```

Set up the environment by copying `.env.example` to `.env.local` and filling in the required values:

- `LIVEKIT_URL`: Use [LiveKit Cloud](https://cloud.livekit.io/) or [run your own](https://docs.livekit.io/home/self-hosting/)
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `DEEPGRAM_API_KEY`: [Get a key](https://console.deepgram.com/) or use your [preferred STT provider](https://docs.livekit.io/agents/integrations/stt/)

You can load the LiveKit environment automatically using the [LiveKit CLI](https://docs.livekit.io/home/cli/cli-setup):

```bash
lk app env -w .env.local
```

## Run the agent

Before your first run, you must download certain models such as [Silero VAD](https://docs.livekit.io/agents/build/turns/vad/) and the [LiveKit turn detector](https://docs.livekit.io/agents/build/turns/turn-detector/):

```console
uv run python src/agent.py download-files
```

Next, run this command to speak to your agent directly in your terminal:

```console
uv run python src/agent.py console
```

To run the agent for use with a frontend or telephony, use the `dev` command:

```console
uv run python src/agent.py dev
```

In production, use the `start` command:

```console
uv run python src/agent.py start
```


This project includes a complete suite of evals, based on the LiveKit Agents [testing & evaluation framework](https://docs.livekit.io/agents/build/testing/). To run them, use `pytest`.

```console
uv run pytest
```

## Using this template repo for your own project



1. **Check in your `uv.lock`**: This file is currently untracked for the template, but you should commit it to your repository for reproducible builds and proper configuration management. (The same applies to `livekit.toml`, if you run your agents in LiveKit Cloud)

2. **Remove the git tracking test**: Delete the "Check files not tracked in git" step from `.github/workflows/tests.yml` since you'll now want this file to be tracked. These are just there for development purposes in the template repo itself.

3. **Add your own repository secrets**: You must [add secrets](https://docs.github.com/en/actions/how-tos/writing-workflows/choosing-what-your-workflow-does/using-secrets-in-github-actions) for `OPENAI_API_KEY` or your other LLM provider so that the tests can run in CI.

## Deploying to production

This project is production-ready and includes a working `Dockerfile`. To deploy it to LiveKit Cloud or another environment, see the [deploying to production](https://docs.livekit.io/agents/ops/deployment/) guide.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
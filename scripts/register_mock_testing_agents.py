import asyncio
import logging
import os
from pathlib import Path

from chungoid.utils.agent_registry import AgentRegistry, AgentCard
from chungoid.runtime.agents.mocks.testing_mock_agents import get_all_mock_testing_agent_cards

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration for ChromaDB
# AgentRegistry uses a hardcoded collection name "a2a_agent_registry"
# and constructs path from project_root and chroma_mode.

async def register_agents(registry: AgentRegistry):
    """Registers all mock testing agents, overwriting if they exist."""
    agent_cards = get_all_mock_testing_agent_cards()
    logger.info(f"Found {len(agent_cards)} mock testing agent cards to register.")

    for card in agent_cards:
        try:
            logger.info(f"Registering agent: {card.agent_id} - {card.name} (overwrite=True)")
            # AgentRegistry.add is synchronous, needs to be run in thread for async context
            await asyncio.to_thread(registry.add, card, overwrite=True) 
            logger.info(f"Successfully registered agent: {card.agent_id}")
        except Exception as e:
            logger.error(f"Failed to register agent {card.agent_id}: {e}", exc_info=True)

    # Optionally, list agents after registration to confirm
    try:
        # AgentRegistry.list is synchronous
        registered_agents = await asyncio.to_thread(registry.list)
        logger.info(f"Currently registered agents ({len(registered_agents)}):")
        for agent_card_obj in registered_agents: # Iterate over AgentCard objects
            logger.info(f"  - ID: {agent_card_obj.agent_id}, Name: {agent_card_obj.name}, Categories: {agent_card_obj.categories}")
    except Exception as e:
        logger.error(f"Failed to list agents after registration: {e}", exc_info=True)


async def main():
    logger.info("Initializing AgentRegistry for dummy_project...")
    # Assumes script is run from workspace root.
    # Point project_root to the dummy_project so agents are registered there.
    dummy_project_path = Path("./dummy_project") # TODO: Make this configurable or use chungoid-core context
    try:
        registry = AgentRegistry(
            project_root=dummy_project_path, 
            chroma_mode="persistent"
        )
        logger.info(f"AgentRegistry initialized for project_root: {dummy_project_path.resolve()}")
        
        await register_agents(registry)
        
    except Exception as e:
        logger.error(f"An error occurred during the agent registration process: {e}", exc_info=True)
        logger.error("Please ensure ChromaDB is running and configured correctly, or that the path is valid for on-disk storage.")

if __name__ == "__main__":
    logger.info("Starting mock testing agent registration script...")
    asyncio.run(main())
    logger.info("Mock testing agent registration script finished.") 
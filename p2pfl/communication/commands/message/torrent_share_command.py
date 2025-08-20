#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Torrent Share Command - Broadcasts torrent files to other nodes."""

import base64
import os
from pathlib import Path
from typing import Optional

from p2pfl.communication.commands.command import Command
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState


class TorrentShareCommand(Command):
    """Command to share torrent files between nodes."""

    def __init__(self, state: NodeState) -> None:
        """
        Initialize the command.
        
        Args:
            state: The node state for accessing node information.
        """
        self.__state = state

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "torrent_share"

    def execute(
        self, 
        source: str, 
        round: int, 
        torrent_data: Optional[str] = None,
        filename: Optional[str] = None,
        info_hash: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Execute the command - receive and save torrent file.

        Args:
            source: The source node address that sent the torrent.
            round: The round number of the torrent.
            torrent_data: Base64 encoded torrent file data.
            filename: Original filename of the torrent.
            info_hash: Info hash of the torrent for logging.
            **kwargs: Additional arguments.

        """
        if torrent_data is None or filename is None:
            logger.warning(self.__state.addr, f"⚠️ Received invalid torrent share from {source}")
            return

        try:
            # Decode base64 torrent data
            torrent_bytes = base64.b64decode(torrent_data)
            
            # Create directory structure: tmp/torrents_{node_name}/
            source_node_name = source.replace(":", "_").replace(".", "_")
            torrent_dir = Path(os.getcwd()) / "tmp" / f"torrents_{source_node_name}"
            torrent_dir.mkdir(parents=True, exist_ok=True)
            
            # Save torrent file with naming: {source_node}_round_{round}.torrent
            torrent_filename = f"{source_node_name}_round_{round}.torrent"
            torrent_path = torrent_dir / torrent_filename
            
            # Write torrent file
            with open(torrent_path, 'wb') as f:
                f.write(torrent_bytes)
            
            # Log success
            hash_info = f" (hash: {info_hash[:8]}...)" if info_hash else ""
            logger.info(
                self.__state.addr, 
                f"📥 Received torrent from {source}: {torrent_filename}{hash_info}"
            )
            
        except Exception as e:
            logger.warning(
                self.__state.addr, 
                f"⚠️ Failed to save torrent from {source}: {str(e)}"
            )
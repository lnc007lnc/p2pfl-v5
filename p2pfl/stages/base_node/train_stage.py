#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
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
"""Train stage."""

import os
from pathlib import Path
from typing import Any, List, Optional, Set, Type, Union

try:
    import libtorrent as lt
    LIBTORRENT_AVAILABLE = True
except ImportError:
    LIBTORRENT_AVAILABLE = False
    logger.warning("libtorrent not available, torrent generation will be skipped")

from p2pfl.communication.commands.message.metrics_command import MetricsCommand
from p2pfl.communication.commands.message.models_agregated_command import ModelsAggregatedCommand
from p2pfl.communication.commands.message.models_ready_command import ModelsReadyCommand
from p2pfl.communication.commands.weights.partial_model_command import PartialModelCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import EarlyStopException, Stage, check_early_stop
from p2pfl.stages.stage_factory import StageFactory


class TrainStage(Stage):
    """Train stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "TrainStage"

    @staticmethod
    def execute(
        state: Optional[NodeState] = None,
        communication_protocol: Optional[CommunicationProtocol] = None,
        learner: Optional[Learner] = None,
        aggregator: Optional[Aggregator] = None,
        **kwargs,
    ) -> Union[Type["Stage"], None]:
        """Execute the stage."""
        if state is None or communication_protocol is None or aggregator is None or learner is None:
            raise Exception("Invalid parameters on TrainStage.")

        try:
            check_early_stop(state)

            # Set Models To Aggregate
            aggregator.set_nodes_to_aggregate(state.train_set)

            check_early_stop(state)

            # Evaluate and send metrics
            TrainStage.__evaluate(state, learner, communication_protocol)

            check_early_stop(state)

            # Train
            logger.info(state.addr, "🏋️‍♀️ Training...")
            learner.fit()
            logger.info(state.addr, "🎓 Training done.")
            
            # Save model weights to tmp folder
            TrainStage.__save_model_weights(state, learner)

            check_early_stop(state)

            # Aggregate Model
            models_added = aggregator.add_model(learner.get_model())

            # send model added msg ---->> redundant (a node always owns its model)
            # TODO: print("Broadcast redundante")
            communication_protocol.broadcast(
                communication_protocol.build_msg(
                    ModelsAggregatedCommand.get_name(),
                    models_added,
                    round=state.round,
                )
            )
            TrainStage.__gossip_model_aggregation(state, communication_protocol, aggregator)

            check_early_stop(state)

            # Set aggregated model
            agg_model = aggregator.wait_and_get_aggregation()
            learner.set_model(agg_model)

            # Share that aggregation is done
            communication_protocol.broadcast(communication_protocol.build_msg(ModelsReadyCommand.get_name(), [], round=state.round))

            # Next stage
            return StageFactory.get_stage("GossipModelStage")
        except EarlyStopException:
            return None

    @staticmethod
    def __save_model_weights(state: NodeState, learner: Learner) -> None:
        """
        Save model weights to tmp/local_model folder after training.
        
        Args:
            state: The node state containing node address and round information.
            learner: The learner containing the trained model.
        """
        try:
            # Create directory structure (configurable via environment variable)
            # Default to local tmp folder in project directory
            default_path = os.path.join(os.getcwd(), "tmp", "local_model")
            base_path = os.environ.get("P2PFL_MODEL_SAVE_PATH", default_path)
            base_dir = Path(base_path)
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename: node_address_round_X.pth
            node_name = state.addr.replace(":", "_").replace(".", "_")  # Replace special chars
            if state.round is not None:
                filename = f"{node_name}_round_{state.round}.pth"
            else:
                filename = f"{node_name}_initial.pth"
            
            filepath = base_dir / filename
            
            # Get model and save weights
            model = learner.get_model()
            encoded_params = model.encode_parameters()
            
            # Save the encoded parameters to file
            with open(filepath, 'wb') as f:
                f.write(encoded_params)
            
            logger.info(state.addr, f"💾 Model weights saved to: {filepath}")
            
            # Generate torrent for the saved weight file
            TrainStage.__generate_torrent(state, filepath)
            
        except Exception as e:
            logger.warning(state.addr, f"⚠️ Failed to save model weights: {str(e)}")

    @staticmethod
    def __generate_torrent(state: NodeState, filepath: Path) -> None:
        """
        Generate a torrent file for the saved weight file.
        
        Args:
            state: The node state containing node address.
            filepath: Path to the weight file to create torrent for.
        """
        if not LIBTORRENT_AVAILABLE:
            logger.debug(state.addr, "Skipping torrent generation (libtorrent not available)")
            return
            
        try:
            # Create torrent directory
            torrent_dir = filepath.parent / "torrents"
            torrent_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate torrent filename
            torrent_path = torrent_dir / f"{filepath.stem}.torrent"
            
            # Create torrent
            fs = lt.file_storage()
            lt.add_files(fs, str(filepath))
            
            # Create torrent with piece size of 256KB
            t = lt.create_torrent(fs, piece_size=256 * 1024)
            
            # Add trackers (you can customize these)
            trackers = [
                "udp://tracker.openbittorrent.com:80/announce",
                "udp://tracker.opentrackr.org:1337/announce",
                "udp://tracker.coppersurfer.tk:6969/announce"
            ]
            for tracker in trackers:
                t.add_tracker(tracker, 0)
            
            # Set creator and comment
            t.set_creator(f"P2PFL Node {state.addr}")
            t.set_comment(f"Model weights for round {state.round}")
            
            # Add web seeds (optional, for HTTP fallback)
            # t.add_url_seed(f"http://your-server.com/models/{filepath.name}")
            
            # Generate torrent
            lt.set_piece_hashes(t, str(filepath.parent))
            torrent_data = lt.bencode(t.generate())
            
            # Save torrent file
            with open(torrent_path, 'wb') as f:
                f.write(torrent_data)
            
            # Calculate info hash for logging
            torrent_info = lt.torrent_info(torrent_data)
            info_hash = str(torrent_info.info_hash())
            
            logger.info(state.addr, f"🌊 Torrent created: {torrent_path.name} (hash: {info_hash[:8]}...)")
            
        except Exception as e:
            logger.warning(state.addr, f"⚠️ Failed to generate torrent: {str(e)}")

    @staticmethod
    def __evaluate(state: NodeState, learner: Learner, communication_protocol: CommunicationProtocol) -> None:
        logger.info(state.addr, "🔬 Evaluating...")
        results = learner.evaluate()
        logger.info(state.addr, f"📈 Evaluated. Results: {results}")
        # Send metrics
        if len(results) > 0:
            logger.info(state.addr, "📢 Broadcasting metrics.")
            flattened_metrics = [str(item) for pair in results.items() for item in pair]
            communication_protocol.broadcast(
                communication_protocol.build_msg(
                    MetricsCommand.get_name(),
                    flattened_metrics,
                    round=state.round,
                )
            )

    @staticmethod
    def __gossip_model_aggregation(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        aggregator: Aggregator,
    ) -> None:
        """
        Gossip model aggregation.

        CAREFULL:
            - Full connected trainset to increase aggregation speed. On real scenarios, this won't
            be possible, private networks and firewalls.
            - Needed because the trainset can split the networks (and neighbors that are not in the
            trainset won't receive the aggregation).
        """

        # Anonymous functions
        def early_stopping_fn():
            return state.round is None

        def get_candidates_fn() -> List[str]:
            candidates = set(state.train_set) - {state.addr}
            return [n for n in candidates if len(TrainStage.__get_remaining_nodes(n, state)) != 0]

        def status_fn() -> Any:
            return [
                (
                    n,
                    TrainStage.__get_aggregated_models(n, state),
                )  # reemplazar por Aggregator - borrarlo de node
                for n in communication_protocol.get_neighbors(only_direct=False)
                if (n in state.train_set)
            ]

        def model_fn(node: str) -> tuple[Any, str, int, list[str]]:
            if state.round is None:
                raise Exception("Round not initialized.")
            try:
                model = aggregator.get_model(TrainStage.__get_aggregated_models(node, state))
            except NoModelsToAggregateError:
                logger.debug(state.addr, f"❔ No models to aggregate for {node}.")
                return (
                    None,
                    PartialModelCommand.get_name(),
                    state.round,
                    [],
                )
            model_msg = communication_protocol.build_weights(
                PartialModelCommand.get_name(),
                state.round,
                model.encode_parameters(),
                model.get_contributors(),
                model.get_num_samples(),
            )
            return (
                model_msg,
                PartialModelCommand.get_name(),
                state.round,
                model.get_contributors(),
            )

        # Gossip
        communication_protocol.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
            create_connection=True,
        )

    @staticmethod
    def __get_aggregated_models(node: str, state: NodeState) -> List[str]:
        try:
            return state.models_aggregated[node]
        except KeyError:
            return []

    @staticmethod
    def __get_remaining_nodes(node: str, state: NodeState) -> Set[str]:
        return set(state.train_set) - set(TrainStage.__get_aggregated_models(node, state))

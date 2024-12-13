package ac.osaka_u.ics.pbl.common

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
sealed class TaskParameter {
    @Serializable
    @SerialName("supervised")
    data class SupervisedParameter(val playerId:String,val games: List<String>) : TaskParameter()
    @Serializable
    @SerialName("reinforcement")
    data class ReinforcementParameter(val numEpoch: Int) : TaskParameter()

}
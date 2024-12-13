package ac.osaka_u.ics.pbl.data.dao

import ac.osaka_u.ics.pbl.common.TaskType
import org.jetbrains.exposed.dao.id.UUIDTable

object Tasks : UUIDTable("tasks") {
    val completed = bool("completed").default(false)
    val baseModelId = reference("base_model_id", Models.id).nullable()
    val type = enumeration("type", TaskType::class)
    val parameters = text("parameters")
}
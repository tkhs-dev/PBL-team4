package ac.osaka_u.ics.pbl.data.dao

import ac.osaka_u.ics.pbl.common.ParameterMapSerializer
import ac.osaka_u.ics.pbl.common.TaskStatus
import ac.osaka_u.ics.pbl.common.TaskType
import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.json.Json
import org.jetbrains.exposed.dao.id.UUIDTable
import org.jetbrains.exposed.sql.json.jsonb
import org.jetbrains.exposed.sql.kotlin.datetime.timestamp

val format = Json { prettyPrint = true }

object Tasks : UUIDTable("tasks") {
    val status = enumeration("status", TaskStatus::class)
    val errorCount = integer("error_count").default(0)
    val baseModelId = reference("base_model_id", Models.id).nullable()
    val type = enumeration("type", TaskType::class)
    val createdAt = timestamp("created_at")
    val parameters = jsonb<Map<String,Any>>("parameters", format, ParameterMapSerializer)
}
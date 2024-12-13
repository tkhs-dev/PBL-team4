package ac.osaka_u.ics.pbl.data.dao

import org.jetbrains.exposed.dao.id.UUIDTable
import org.jetbrains.exposed.sql.kotlin.datetime.timestamp

object Models: UUIDTable("models") {
    val version = integer("version")
    val parentId = reference("parent_id", Models.id).nullable()
    val rootModel = reference("root_model", Models.id).nullable()
    val sequence = integer("sequence")
    val taskId = reference("task_id", Tasks.id)
    val createdAt = timestamp("created_at")
}
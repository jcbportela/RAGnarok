/**
 * Skill File Manager — generates and manages SKILL.md files
 * for Copilot agent discovery of RAG topics.
 *
 * Output location: ~/.copilot/skills/ (auto-discovered by Copilot)
 */

import * as vscode from 'vscode';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import { Topic } from '../utils/types';
import { CONFIG } from '../utils/constants';
import { Logger } from '../utils/logger';

const logger = new Logger('SkillFileManager');

/** Prefix used for generated skill directory names */
const SKILL_DIR_PREFIX = 'rag-';

/** Marker comment embedded in generated SKILL.md to identify auto-generated files */
const GENERATED_MARKER = '<!-- generated-by: ragnarok -->';

/** Global state key storing the set of topic IDs with individual skill generation enabled */
const STATE_TOPIC_SKILL_OVERRIDES = 'ragnarok.topicSkillOverrides';

export class SkillFileManager {
  private extensionPath: string;
  private templateContent: string | null = null;

  constructor(private context: vscode.ExtensionContext) {
    this.extensionPath = context.extensionPath;
  }

  // ── Public API ─────────────────────────────────────────────

  /**
   * Generate a SKILL.md for a topic.
   * When the global setting is enabled, all topics get skills.
   * When disabled, only topics with individual overrides get skills.
   */
  public async generateSkillFile(topic: Topic): Promise<void> {
    if (!this.shouldGenerateForTopic(topic)) {
      return;
    }

    const slug = this.toSlug(topic.name);
    const skillDir = await this.ensureSkillDirectory(slug);
    const content = await this.renderTemplate(topic);

    await fs.writeFile(path.join(skillDir, 'SKILL.md'), content, 'utf-8');
    logger.info(`Skill file generated for topic "${topic.name}" at ${skillDir}`);
  }

  /**
   * Delete the SKILL.md for a topic unconditionally.
   */
  public async deleteSkillFile(topicName: string): Promise<void> {
    const slug = this.toSlug(topicName);
    const skillDir = this.getSkillDirPath(slug);

    try {
      await fs.rm(skillDir, { recursive: true, force: true });
      logger.info(`Skill file deleted for topic "${topicName}" at ${skillDir}`);
    } catch {
      // Directory may not exist, that's fine
    }
  }

  /**
   * Handle a topic rename: delete the old skill, generate the new one.
   */
  public async renameSkillFile(oldName: string, topic: Topic): Promise<void> {
    if (!this.shouldGenerateForTopic(topic)) {
      // Still delete the old one even if new one won't be generated
      await this.deleteSkillFile(oldName);
      return;
    }

    const oldSlug = this.toSlug(oldName);
    const newSlug = this.toSlug(topic.name);

    // Only act if the slug actually changed
    if (oldSlug !== newSlug) {
      const oldDir = this.getSkillDirPath(oldSlug);
      try {
        await fs.rm(oldDir, { recursive: true, force: true });
      } catch {
        // may not exist
      }
    }

    await this.generateSkillFile(topic);
  }

  /**
   * Generate skill files for all provided topics (used when the feature is toggled ON).
   * Bypasses the shouldGenerateForTopic check — caller is responsible for deciding when to call this.
   */
  public async generateAllSkillFiles(topics: Topic[]): Promise<void> {
    for (const topic of topics) {
      const slug = this.toSlug(topic.name);
      const skillDir = await this.ensureSkillDirectory(slug);
      const content = await this.renderTemplate(topic);
      await fs.writeFile(path.join(skillDir, 'SKILL.md'), content, 'utf-8');
    }
    logger.info(`Generated skill files for ${topics.length} topic(s)`);
  }

  /**
   * Remove ALL generated skill files and clear per-topic overrides
   * (used when the feature is toggled OFF).
   */
  public async removeAllSkillFiles(topics: Topic[]): Promise<void> {
    for (const topic of topics) {
      await this.deleteSkillFile(topic.name);
    }
    // Clear per-topic overrides so UI state is consistent
    await this.context.globalState.update(STATE_TOPIC_SKILL_OVERRIDES, []);
    logger.info(`Removed all generated skill files and cleared per-topic overrides`);
  }

  /**
   * Reconcile skill files at startup: generate any missing files for topics
   * that should have them (based on global setting + per-topic overrides).
   */
  public async reconcile(topics: Topic[]): Promise<void> {
    let generated = 0;
    for (const topic of topics) {
      if (!this.shouldGenerateForTopic(topic)) {
        continue;
      }
      const slug = this.toSlug(topic.name);
      const skillFile = path.join(this.getSkillDirPath(slug), 'SKILL.md');
      try {
        await fs.access(skillFile);
      } catch {
        // File doesn't exist — generate it
        const skillDir = await this.ensureSkillDirectory(slug);
        const content = await this.renderTemplate(topic);
        await fs.writeFile(path.join(skillDir, 'SKILL.md'), content, 'utf-8');
        generated++;
      }
    }
    if (generated > 0) {
      logger.info(`Reconciled ${generated} missing skill file(s)`);
    }
  }

  // ── Per-topic skill overrides ──────────────────────────────

  /**
   * Check whether an individual topic has its skill override enabled.
   * Only meaningful when the global setting is disabled.
   */
  public isTopicSkillEnabled(topicId: string): boolean {
    const overrides = this.context.globalState.get<string[]>(STATE_TOPIC_SKILL_OVERRIDES, []);
    return overrides.includes(topicId);
  }

  /**
   * Toggle the per-topic skill override.
   * When toggled ON and the global setting is OFF, generates the skill file.
   * When toggled OFF, removes the skill file.
   * Returns the new toggle state.
   */
  public async toggleTopicSkill(topic: Topic): Promise<boolean> {
    const overrides = this.context.globalState.get<string[]>(STATE_TOPIC_SKILL_OVERRIDES, []);
    const idx = overrides.indexOf(topic.id);
    const wasEnabled = idx !== -1;

    if (wasEnabled) {
      overrides.splice(idx, 1);
      await this.context.globalState.update(STATE_TOPIC_SKILL_OVERRIDES, overrides);
      await this.deleteSkillFile(topic.name);
      logger.info(`Per-topic skill disabled for "${topic.name}"`);
    } else {
      overrides.push(topic.id);
      await this.context.globalState.update(STATE_TOPIC_SKILL_OVERRIDES, overrides);
      // Force-generate regardless of global setting
      const slug = this.toSlug(topic.name);
      const skillDir = await this.ensureSkillDirectory(slug);
      const content = await this.renderTemplate(topic);
      await fs.writeFile(path.join(skillDir, 'SKILL.md'), content, 'utf-8');
      logger.info(`Per-topic skill enabled for "${topic.name}"`);
    }

    return !wasEnabled;
  }

  /**
   * (Re)generate skill files.
   * Considers global setting + per-topic overrides.
   */
  public async regenerateSkillFiles(topics: Topic[]): Promise<void> {
    const globalEnabled = this.isEnabled();

    for (const topic of topics) {
      if (globalEnabled || this.isTopicSkillEnabled(topic.id)) {
        const slug = this.toSlug(topic.name);
        const skillDir = await this.ensureSkillDirectory(slug);
        const content = await this.renderTemplate(topic);
        await fs.writeFile(path.join(skillDir, 'SKILL.md'), content, 'utf-8');
        logger.info(`Skill file (re)generated for topic "${topic.name}"`);
      } else {
        // Make sure there's no stale skill file
        await this.deleteSkillFile(topic.name);
      }
    }

    logger.info(`Skill files regenerated for ${topics.length} topic(s)`);
  }

  // ── Configuration helpers ──────────────────────────────────

  /**
   * Determine whether a particular topic should have a skill file generated.
   */
  private shouldGenerateForTopic(topic: Topic): boolean {
    if (this.isEnabled()) {
      return true;
    }
    // Global setting is off — check per-topic override
    return this.isTopicSkillEnabled(topic.id);
  }

  private isEnabled(): boolean {
    return vscode.workspace
      .getConfiguration(CONFIG.ROOT)
      .get<boolean>(CONFIG.GENERATE_SKILL_FILES, true);
  }

  // ── Path helpers ───────────────────────────────────────────

  /**
   * Get the base skills directory (~/.copilot/skills/).
   */
  private getBaseSkillsDir(): string {
    return path.join(os.homedir(), '.copilot', 'skills');
  }

  /**
   * Full path for a specific skill directory.
   */
  private getSkillDirPath(slug: string): string {
    return path.join(this.getBaseSkillsDir(), slug);
  }

  /**
   * Ensure the skill directory exists and return its path.
   */
  private async ensureSkillDirectory(slug: string): Promise<string> {
    const dir = this.getSkillDirPath(slug);
    await fs.mkdir(dir, { recursive: true });
    return dir;
  }

  // ── Template rendering ─────────────────────────────────────

  /**
   * Load the bundled template (cached after first load).
   */
  private async loadTemplate(): Promise<string> {
    if (this.templateContent) {
      return this.templateContent;
    }
    const templatePath = path.join(this.extensionPath, 'assets', 'templates', 'skill-template.md');
    this.templateContent = await fs.readFile(templatePath, 'utf-8');
    return this.templateContent;
  }

  /**
   * Render the template with topic data.
   */
  private async renderTemplate(topic: Topic): Promise<string> {
    const template = await this.loadTemplate();

    const slug = this.toSlug(topic.name);
    const description = topic.description || '';
    const source = topic.source || 'local';

    let rendered = template
      .replace(/\{\{topicSlug\}\}/g, slug)
      .replace(/\{\{topicName\}\}/g, topic.name)
      .replace(/\{\{topicDescription\}\}/g, description)
      .replace(/\{\{documentCount\}\}/g, String(topic.documentCount))
      .replace(/\{\{topicSource\}\}/g, source);

    // Handle conditional description block: {{#topicDescription}} ... {{/topicDescription}}
    if (description) {
      rendered = rendered
        .replace(/\{\{#topicDescription\}\}/g, '')
        .replace(/\{\{\/topicDescription\}\}/g, '');
    } else {
      rendered = rendered.replace(
        /\{\{#topicDescription\}\}[\s\S]*?\{\{\/topicDescription\}\}/g,
        ''
      );
    }

    // Append the generated marker as the very last line
    rendered = rendered.trimEnd() + '\n\n' + GENERATED_MARKER + '\n';

    return rendered;
  }



  // ── Utilities ──────────────────────────────────────────────

  /**
   * Convert a topic name to a URL/directory-safe kebab-case slug.
   * The slug is prefixed with "rag-" to namespace generated skill directories.
   */
  private toSlug(name: string): string {
    const base = name
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9\s-]/g, '')  // remove non-alphanumeric
      .replace(/[\s_]+/g, '-')        // spaces/underscores → hyphens
      .replace(/-+/g, '-')            // collapse multiple hyphens
      .replace(/^-|-$/g, '');         // trim leading/trailing hyphens
    return `${SKILL_DIR_PREFIX}${base}`;
  }

}
